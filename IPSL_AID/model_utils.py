# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
import datetime
import os


class ModelUtils:
    """
    Utility class for model inspection, checkpointing, and memory profiling.

    This class provides static methods for common model operations including
    parameter counting, memory usage analysis, checkpoint management, and
    model inspection.

    Examples
    --------
    >>> utils = ModelUtils()
    >>> param_counts = ModelUtils.get_parameter_number(model)
    >>> ModelUtils.save_checkpoint(state, "checkpoint.pth.tar", logger)
    """

    def __init__(self):
        """Initialize ModelUtils instance."""
        pass

    @staticmethod
    def get_parameter_number(model, logger=None):
        """
        Calculate the total and trainable number of parameters in a model.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model to inspect
        logger : Logger, optional
            Logger instance for output, by default None

        Returns
        -------
        dict
            Dictionary containing:
            - 'Total': Total number of parameters
            - 'Trainable': Number of trainable parameters

        Examples
        --------
        >>> model = torch.nn.Linear(10, 5)
        >>> counts = ModelUtils.get_parameter_number(model, logger)
        """
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if logger:
            logger.info(
                f"Model Parameters - Total: {total_num:,}, Trainable: {trainable_num:,}"
            )

        return {"Total": total_num, "Trainable": trainable_num}

    @staticmethod
    def print_model_layers(model, logger=None):
        """
        Print model parameter names along with their gradient requirements.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model to inspect
        logger : Logger, optional
            Logger instance for output, by default None

        Examples
        --------
        >>> model = torch.nn.Sequential(
        ...     torch.nn.Linear(10, 5),
        ...     torch.nn.ReLU(),
        ...     torch.nn.Linear(5, 1)
        ... )
        >>> ModelUtils.print_model_layers(model, logger)
        """
        if logger:
            logger.info("Model Layer Information:")
            for name, param in model.named_parameters():
                logger.info(f"  Layer: {name}, Requires Grad: {param.requires_grad}")
        else:
            for name, param in model.named_parameters():
                print(f"Layer: {name},\t Requires Grad: {param.requires_grad}")

    @staticmethod
    def save_checkpoint(state, filename="checkpoint.pth.tar", logger=None):
        """
        Save model and optimizer state to a file.

        Parameters
        ----------
        state : dict
            Dictionary containing model state_dict and other training information.
            Typically includes:
            - 'state_dict': Model parameters
            - 'optimizer': Optimizer state
            - 'epoch': Current epoch
            - 'loss': Current loss value
        filename : str, optional
            File path to save the checkpoint, by default "checkpoint.pth.tar"
        logger : Logger, optional
            Logger instance for output, by default None

        Examples
        --------
        >>> state = {
        ...     'state_dict': model.state_dict(),
        ...     'optimizer': optimizer.state_dict(),
        ...     'epoch': epoch,
        ...     'loss': loss
        ... }
        >>> ModelUtils.save_checkpoint(state, 'model_checkpoint.pth.tar', logger)
        """
        if logger:
            logger.info(f"Saving checkpoint to: {filename}")
        else:
            print(f"=> Saving checkpoint to: {filename}")
        torch.save(state, filename)

    @staticmethod
    def load_checkpoint(checkpoint, model, optimizer=None, logger=None):
        """
        Load model and optimizer state from a checkpoint file.

        Parameters
        ----------
        checkpoint : dict
            Loaded checkpoint dictionary
        model : torch.nn.Module
            Model to load weights into
        optimizer : torch.optim.Optimizer, optional
            Optimizer to restore state, by default None
        logger : Logger, optional
            Logger instance for output, by default None

        Examples
        --------
        >>> checkpoint = torch.load('model_checkpoint.pth.tar')
        >>> ModelUtils.load_checkpoint(checkpoint, model, optimizer, logger)
        """
        if logger:
            logger.info("Loading checkpoint")
        else:
            print("=> Loading checkpoint")

        model.load_state_dict(checkpoint["state_dict"])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if logger:
                logger.info("Optimizer state restored")

        if logger:
            logger.info("Checkpoint loaded successfully")

    @staticmethod
    def load_training_checkpoint(
        checkpoint_path, model, optimizer, device, logger=None
    ):
        """
        Load comprehensive training checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint file
        model : torch.nn.Module
            Model to load weights into
        optimizer : torch.optim.Optimizer
            Optimizer to restore state
        device : torch.device
            Device to load checkpoint to
        logger : Logger, optional
            Logger instance for output

        Returns
        -------
        tuple
            (epoch, samples_processed, batches_processed, best_val_loss, best_epoch, checkpoint)
        """
        if not os.path.exists(checkpoint_path):
            if logger:
                logger.error(f"Checkpoint not found at: {checkpoint_path}")
            return None, 0, 0, float("inf"), 0, None

        if logger:
            logger.info(f"Loading checkpoint from: '{checkpoint_path}'")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        if logger:
            logger.info("Checkpoint loaded into memory")
            logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

        # Handle DataParallel compatibility
        if torch.cuda.device_count() > 1 and isinstance(model, torch.nn.DataParallel):
            # Check if checkpoint was saved from DataParallel
            first_key = next(iter(checkpoint["state_dict"].keys()))
            if not first_key.startswith("module."):
                # Wrap state dict with 'module.' prefix for DataParallel
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for k, v in checkpoint["state_dict"].items():
                    new_state_dict["module." + k] = v
                checkpoint["state_dict"] = new_state_dict

        # Load model and optimizer states
        ModelUtils.load_checkpoint(checkpoint, model, optimizer, logger=logger)

        # Extract training state
        epoch = checkpoint.get("epoch", 0)
        samples_processed = checkpoint.get("samples_processed", 0)
        batches_processed = checkpoint.get("batches_processed", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_epoch = checkpoint.get("best_epoch", 0)

        if logger:
            logger.info(
                f"Checkpoint loaded: epoch {epoch}, {samples_processed:,} samples"
            )
            logger.info("Training state extracted:")
            logger.info(f" └── epoch: {epoch}")
            logger.info(f" └── samples_processed: {samples_processed}")
            logger.info(f" └── batches_processed: {batches_processed}")
            logger.info(f" └── best_val_loss: {best_val_loss}")
            logger.info(f" └── best_epoch: {best_epoch}")

        return (
            epoch,
            samples_processed,
            batches_processed,
            best_val_loss,
            best_epoch,
            checkpoint,
        )

    @staticmethod
    def count_parameters_by_layer(model, logger=None):
        """
        Count parameters for each layer in the model.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model to analyze
        logger : Logger, optional
            Logger instance for output, by default None

        Returns
        -------
        dict
            Dictionary with layer names as keys and parameter counts as values

        Examples
        --------
        >>> layer_params = ModelUtils.count_parameters_by_layer(model, logger)
        """
        layer_params = {}
        for name, param in model.named_parameters():
            layer_params[name] = param.numel()

        if logger:
            logger.info("Parameter count by layer:")
            for layer, count in layer_params.items():
                logger.info(f"  {layer}: {count:,} parameters")

        return layer_params

    @staticmethod
    def log_model_summary(model, input_shape=None, logger=None):
        """
        Log comprehensive model summary including parameters and architecture.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model to summarize
        input_shape : tuple, optional
            Input shape for memory analysis, by default None
        logger : Logger, optional
            Logger instance for output, by default None
        """
        if logger:
            logger.info("=" * 60)
            logger.info("MODEL SUMMARY")
            logger.info("=" * 60)

            # Parameter counts
            param_counts = ModelUtils.get_parameter_number(model, logger=None)
            logger.info(f"Total Parameters: {param_counts['Total']:,}")
            logger.info(f"Trainable Parameters: {param_counts['Trainable']:,}")

            # Layer information
            logger.info("\nLayer Details:")
            ModelUtils.print_model_layers(model, logger)

            logger.info("=" * 60)

    @staticmethod
    def save_training_checkpoint(
        model,
        optimizer,
        epoch,
        samples_processed,
        batches_processed,
        train_loss_history,
        valid_loss_history,
        valid_metrics_history,
        best_val_loss,
        best_epoch,
        avg_val_loss,
        avg_epoch_loss,
        args,
        paths,
        logger,
        checkpoint_type="epoch",
        save_full_model=True,
    ):
        """
        Save comprehensive training checkpoint with consistent formatting.

        Parameters
        ----------
        model : torch.nn.Module
            Model to save
        optimizer : torch.optim.Optimizer
            Optimizer to save
        epoch : int
            Current epoch
        samples_processed : int
            Number of samples processed so far
        batches_processed : int
            Number of batches processed so far
        train_loss_history : list
            History of training losses
        valid_loss_history : list
            History of validation losses
        valid_metrics_history : dict
            History of validation metrics
        best_val_loss : float
            Best validation loss so far
        best_epoch : int
            Epoch with best validation loss
        avg_val_loss : float
            Current epoch validation loss
        avg_epoch_loss : float
            Current epoch training loss
        args : argparse.Namespace
            Command line arguments
        paths : EasyDict
            Directory paths
        logger : Logger
            Logger instance
        checkpoint_type : str
            Type of checkpoint: "samples", "epoch", "best", "final"
        save_full_model : bool
            Whether to also save the full model separately

        Returns
        -------
        tuple
            (checkpoint_filename, full_model_filename)

        Examples
        --------
        >>> checkpoint_file, full_model_file = ModelUtils.save_training_checkpoint(
        ...     model, optimizer, epoch, samples_processed, batches_processed,
        ...     train_loss_history, valid_loss_history, valid_metrics_history,
        ...     best_val_loss, best_epoch, avg_val_loss, avg_epoch_loss,
        ...     args, paths, logger, checkpoint_type="best"
        ... )
        """

        # Handle DataParallel for state dict
        if torch.cuda.device_count() > 1 and isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        # Base checkpoint state
        checkpoint_state = {
            "epoch": epoch,
            "state_dict": state_dict,
            "optimizer": optimizer.state_dict(),
            "samples_processed": samples_processed,
            "batches_processed": batches_processed,
            "train_loss_history": train_loss_history,
            "valid_loss_history": valid_loss_history,
            "valid_metrics_history": valid_metrics_history,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "val_loss": avg_val_loss,
            "train_loss": avg_epoch_loss,
            "checkpoint_type": checkpoint_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "args": vars(args) if hasattr(args, "__dict__") else args,
        }

        # Determine filename based on checkpoint type
        prefix = getattr(args, "prefix", "run")
        save_checkpoint_name = getattr(args, "save_checkpoint_name", "model")

        if checkpoint_type == "samples":
            checkpoint_filename = os.path.join(
                paths.checkpoints,
                f"{prefix}_epoch{epoch:04d}_samples{samples_processed}_{save_checkpoint_name}.pth.tar",
            )
            full_model_filename = os.path.join(
                paths.checkpoints,
                f"{prefix}_epoch{epoch:04d}_samples{samples_processed}_{save_checkpoint_name}_full.pth",
            )

        elif checkpoint_type == "epoch":
            checkpoint_filename = os.path.join(
                paths.checkpoints,
                f"{prefix}_epoch{epoch:04d}_{save_checkpoint_name}.pth.tar",
            )
            full_model_filename = os.path.join(
                paths.checkpoints,
                f"{prefix}_epoch{epoch:04d}_{save_checkpoint_name}_full.pth",
            )

        elif checkpoint_type == "best":
            checkpoint_filename = os.path.join(
                paths.checkpoints, f"{prefix}_best_model.pth.tar"
            )
            full_model_filename = os.path.join(
                paths.checkpoints, f"{prefix}_best_model_full.pth"
            )

        elif checkpoint_type == "final":
            num_epochs = getattr(args, "num_epochs", epoch + 1)
            checkpoint_filename = os.path.join(
                paths.checkpoints, f"{prefix}_final_model_epoch{num_epochs}.pth.tar"
            )
            full_model_filename = os.path.join(
                paths.checkpoints, f"{prefix}_final_model_epoch{num_epochs}_full.pth"
            )
        elif checkpoint_type.startswith("emergency"):
            checkpoint_filename = os.path.join(
                paths.checkpoints,
                f"{prefix}_{checkpoint_type}_{save_checkpoint_name}.pth.tar",
            )
            full_model_filename = os.path.join(
                paths.checkpoints,
                f"{prefix}_{checkpoint_type}_{save_checkpoint_name}_full.pth",
            )

        else:
            if logger:
                logger.warning(
                    f"Unknown checkpoint_type: {checkpoint_type}, using epoch"
                )
            checkpoint_filename = os.path.join(
                paths.checkpoints,
                f"{prefix}_epoch{epoch:04d}_{save_checkpoint_name}.pth.tar",
            )
            full_model_filename = os.path.join(
                paths.checkpoints,
                f"{prefix}_epoch{epoch:04d}_{save_checkpoint_name}_full.pth",
            )

        # Save checkpoint using existing method
        ModelUtils.save_checkpoint(checkpoint_state, checkpoint_filename, logger=logger)

        # Save full model separately if requested
        if save_full_model:
            if torch.cuda.device_count() > 1 and isinstance(
                model, torch.nn.DataParallel
            ):
                torch.save(model.module, full_model_filename)
            else:
                torch.save(model, full_model_filename)

        # Log information
        if logger:
            if checkpoint_type == "best":
                logger.info(f"✅ Best model saved: {checkpoint_filename}")
                logger.info(f" └── Validation loss: {avg_val_loss:.4f}")
            elif checkpoint_type == "final":
                logger.info(f"✅ Final model saved: {checkpoint_filename}")
                logger.info(
                    f" └── Total samples: {samples_processed:,}, Total batches: {batches_processed:,}"
                )
            else:
                logger.info(f"✅ Checkpoint saved: {checkpoint_filename}")

    @staticmethod
    def save_emergency_checkpoint(
        model,
        optimizer,
        epoch,
        samples_processed,
        batches_processed,
        train_loss_history,
        valid_loss_history,
        valid_metrics_history,
        args,
        paths,
        logger,
        reason="emergency",
    ):
        """
        Save emergency checkpoint for recovery.

        Parameters
        ----------
        reason : str
            Reason for emergency save (e.g., "crash", "interrupt", "error")

        Returns
        -------
        tuple
            (checkpoint_filename, full_model_filename)
        """
        ModelUtils.save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            samples_processed=samples_processed,
            batches_processed=batches_processed,
            train_loss_history=train_loss_history,
            valid_loss_history=valid_loss_history,
            valid_metrics_history=valid_metrics_history,
            best_val_loss=float("inf"),
            best_epoch=0,
            avg_val_loss=0.0,
            avg_epoch_loss=0.0,
            args=args,
            paths=paths,
            logger=logger,
            checkpoint_type=f"emergency_{reason}",
            save_full_model=True,
        )
