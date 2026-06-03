import argparse

import torch
from models.train import train
from models.utils.utils import (
    load_and_log_config,
    load_model_and_tokenizer,
    set_global_seed,
)

from data.load_data import load_data


def main(config_path, device_arg=None):
    config, logger = load_and_log_config(config_path)
    logger.info(f"Training parameters loaded: {config}")

    # Set all global seeds
    set_global_seed(config["seed"])

    # Device - priority: command-line arg > config > auto-detect
    if device_arg is not None:
        device_config = device_arg
    else:
        device_config = config.get("device", "auto")

    if device_config == "auto":
        # Auto-detect best available device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_config)

    logger.info(f"Using device: {device}")

    # Load model and tokenizer
    model_name = config["model"]
    use_random_init = config.get("use_random_init", False)

    if use_random_init:
        # Load tokenizer and model with selective random initialization
        from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
        logger.info(f"Creating model with SELECTIVE RANDOM INITIALIZATION from {model_name}")
        logger.info("Preserving: vocabulary embeddings and output projection layer")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # First load the pretrained model to get the embeddings
        pretrained_model = AutoModelForCausalLM.from_pretrained(model_name)

        # Create a new model with random initialization
        model_config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(model_config)

        # Copy the pretrained embeddings and output layer
        # For GPT-2 models, embeddings are in transformer.wte (word token embeddings) and transformer.wpe (position embeddings)
        # Output layer shares weights with input embeddings in GPT-2 (lm_head.weight is tied to transformer.wte.weight)
        if hasattr(model, 'transformer'):
            # GPT-2 architecture
            with torch.no_grad():
                # Preserve word embeddings
                model.transformer.wte.weight.copy_(pretrained_model.transformer.wte.weight)
                # Preserve position embeddings
                model.transformer.wpe.weight.copy_(pretrained_model.transformer.wpe.weight)
                # For GPT-2, lm_head shares weights with wte, so it's automatically preserved
                logger.info("Preserved: word embeddings (wte), position embeddings (wpe)")
                logger.info("Note: Output projection (lm_head) shares weights with word embeddings in GPT-2")

        # Clean up pretrained model to save memory
        del pretrained_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        model = model.to(device)
        logger.info(f"Model initialized with preserved embeddings from: {model_name}")
    else:
        # Load pretrained model (normal mode)
        use_fp16 = config.get("use_fp16", False)
        model, tokenizer = load_model_and_tokenizer(model_name, device, use_fp16)
        logger.info(f"Model and tokenizer loaded (pretrained): {model_name}")
        if use_fp16:
            logger.info("Using FP16 (half precision) for memory savings")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, eval_loader, mean_rt_train, std_rt_train = load_data(
        config, tokenizer
    )
    logger.info(f"Mean and std of training reading times: {mean_rt_train}, {std_rt_train}")

    # Only load reference model if KL weight > 0
    kl_weight = config["training_kwargs"].get("kl_weight", 0.0)
    if kl_weight > 0 and config["training_kwargs"].get("kl_variant", None) == "full":
        if use_random_init:
            # For control condition, reference model should have the same initialization
            from transformers import AutoConfig, AutoModelForCausalLM
            logger.info("Creating reference model with selective random initialization")

            # Load pretrained model to get embeddings
            pretrained_model = AutoModelForCausalLM.from_pretrained(model_name)

            # Create randomly initialized model
            model_config = AutoConfig.from_pretrained(model_name)
            model_ref = AutoModelForCausalLM.from_config(model_config)

            # Copy embeddings from pretrained model
            if hasattr(model_ref, 'transformer'):
                with torch.no_grad():
                    model_ref.transformer.wte.weight.copy_(pretrained_model.transformer.wte.weight)
                    model_ref.transformer.wpe.weight.copy_(pretrained_model.transformer.wpe.weight)

            del pretrained_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            model_ref = model_ref.to(device)
            logger.info("Reference model initialized with preserved embeddings")
        else:
            model_ref, _ = load_model_and_tokenizer(model_name, device)
            logger.info("Reference model loaded (pretrained)")
    else:
        model_ref = None
    train(
        model,
        tokenizer,
        train_loader,
        eval_loader,
        config,
        logger,
        device,
        model_ref=model_ref,
    )


if __name__ == "__main__":
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="LRDPO")
    parser.add_argument(
        "--config_path", "--config", type=str, help="Path to the configuration file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda:0', 'cuda:1', 'mps', 'cpu'). Overrides config setting. Default: auto-detect",
    )
    args = parser.parse_args()

    main(args.config_path, device_arg=args.device)
