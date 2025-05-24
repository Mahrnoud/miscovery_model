"""
Enhanced Learning Rate Scheduler with Custom Control
"""
import math
import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup


class CustomLRScheduler:
    """
    Custom learning rate scheduler with multiple strategies
    """

    def __init__(self, optimizer, scheduler_type, **kwargs):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type

        if scheduler_type == "constant":
            # Fixed learning rate - no change throughout training
            self.scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=kwargs.get('warmup_steps', 0)
            )

        elif scheduler_type == "linear_decay_to_min":
            # Start from initial LR, decay linearly to min_lr, then stay constant
            self.scheduler = self._create_linear_decay_to_min_scheduler(
                optimizer,
                num_warmup_steps=kwargs.get('warmup_steps', 0),
                num_training_steps=kwargs['num_training_steps'],
                min_lr_ratio=kwargs.get('min_lr_ratio', 0.1),
                decay_start_step=kwargs.get('decay_start_step', 0)
            )

        elif scheduler_type == "cosine_decay_to_min":
            # Start from initial LR, decay with cosine to min_lr, then stay constant
            self.scheduler = self._create_cosine_decay_to_min_scheduler(
                optimizer,
                num_warmup_steps=kwargs.get('warmup_steps', 0),
                num_training_steps=kwargs['num_training_steps'],
                min_lr_ratio=kwargs.get('min_lr_ratio', 0.1),
                decay_start_step=kwargs.get('decay_start_step', 0)
            )

        elif scheduler_type == "step_decay_to_min":
            # Step decay: reduce LR by factor every N steps until min_lr
            self.scheduler = self._create_step_decay_to_min_scheduler(
                optimizer,
                step_size=kwargs.get('step_size', 1000),
                decay_factor=kwargs.get('decay_factor', 0.5),
                min_lr_ratio=kwargs.get('min_lr_ratio', 0.1)
            )

        else:
            # Fallback to cosine with warmup
            self.scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=kwargs.get('warmup_steps', 0),
                num_training_steps=kwargs['num_training_steps']
            )

    def _create_linear_decay_to_min_scheduler(self, optimizer, num_warmup_steps,
                                              num_training_steps, min_lr_ratio, decay_start_step):
        """
        Linear decay from initial LR to min_lr, then constant
        """

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                # Warmup phase
                return float(current_step) / float(max(1, num_warmup_steps))

            if current_step < decay_start_step:
                # Stay at full LR until decay_start_step
                return 1.0

            decay_steps = num_training_steps - decay_start_step
            if decay_steps <= 0:
                return min_lr_ratio

            # Linear decay phase
            progress = (current_step - decay_start_step) / decay_steps
            if progress >= 1.0:
                # After decay is complete, stay at min_lr
                return min_lr_ratio

            # Linear interpolation from 1.0 to min_lr_ratio
            return 1.0 - progress * (1.0 - min_lr_ratio)

        return LambdaLR(optimizer, lr_lambda)

    def _create_cosine_decay_to_min_scheduler(self, optimizer, num_warmup_steps,
                                              num_training_steps, min_lr_ratio, decay_start_step):
        """
        Cosine decay from initial LR to min_lr, then constant
        """

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                # Warmup phase
                return float(current_step) / float(max(1, num_warmup_steps))

            if current_step < decay_start_step:
                # Stay at full LR until decay_start_step
                return 1.0

            decay_steps = num_training_steps - decay_start_step
            if decay_steps <= 0:
                return min_lr_ratio

            # Cosine decay phase
            progress = (current_step - decay_start_step) / decay_steps
            if progress >= 1.0:
                # After decay is complete, stay at min_lr
                return min_lr_ratio

            # Cosine decay from 1.0 to min_lr_ratio
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

        return LambdaLR(optimizer, lr_lambda)

    def _create_step_decay_to_min_scheduler(self, optimizer, step_size, decay_factor, min_lr_ratio):
        """
        Step decay: multiply LR by decay_factor every step_size steps until min_lr
        """

        def lr_lambda(current_step):
            # Calculate how many decay steps have occurred
            decay_count = current_step // step_size
            current_ratio = decay_factor ** decay_count

            # Don't go below min_lr_ratio
            return max(current_ratio, min_lr_ratio)

        return LambdaLR(optimizer, lr_lambda)

    def step(self):
        """Step the scheduler"""
        self.scheduler.step()

    def get_last_lr(self):
        """Get the last learning rate"""
        return self.scheduler.get_last_lr()


def create_custom_scheduler(optimizer, args, total_steps):
    """
    Factory function to create custom learning rate scheduler based on arguments
    """
    scheduler_kwargs = {
        'warmup_steps': args.warmup_steps,
        'num_training_steps': total_steps,
        'min_lr_ratio': getattr(args, 'min_lr_ratio', 0.1),
        'decay_start_step': getattr(args, 'decay_start_step', 0),
        'step_size': getattr(args, 'lr_step_size', 1000),
        'decay_factor': getattr(args, 'lr_decay_factor', 0.5)
    }

    return CustomLRScheduler(
        optimizer=optimizer,
        scheduler_type=args.lr_scheduler_type,
        **scheduler_kwargs
    )


# Example usage and testing function
def test_schedulers():
    """
    Test function to visualize different scheduler behaviors
    """
    import matplotlib.pyplot as plt
    import torch.optim as optim

    # Create a dummy model and optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Test parameters
    total_steps = 5000
    warmup_steps = 100

    # Test different schedulers
    scheduler_configs = [
        {"type": "constant", "label": "Constant LR"},
        {"type": "linear_decay_to_min", "label": "Linear Decay to Min",
         "min_lr_ratio": 0.1, "decay_start_step": 500},
        {"type": "cosine_decay_to_min", "label": "Cosine Decay to Min",
         "min_lr_ratio": 0.1, "decay_start_step": 500},
        {"type": "step_decay_to_min", "label": "Step Decay to Min",
         "step_size": 1000, "decay_factor": 0.5, "min_lr_ratio": 0.1}
    ]

    plt.figure(figsize=(12, 8))

    for config in scheduler_configs:
        # Reset optimizer
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        # Create scheduler
        scheduler_kwargs = {
            'warmup_steps': warmup_steps,
            'num_training_steps': total_steps,
            'min_lr_ratio': config.get('min_lr_ratio', 0.1),
            'decay_start_step': config.get('decay_start_step', 0),
            'step_size': config.get('step_size', 1000),
            'decay_factor': config.get('decay_factor', 0.5)
        }

        scheduler = CustomLRScheduler(
            optimizer=optimizer,
            scheduler_type=config["type"],
            **scheduler_kwargs
        )

        # Track learning rates
        lrs = []
        for step in range(total_steps):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        plt.plot(range(total_steps), lrs, label=config["label"], linewidth=2)

    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Scheduler Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()


if __name__ == "__main__":
    test_schedulers()
