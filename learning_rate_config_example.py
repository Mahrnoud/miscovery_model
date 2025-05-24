# ========================================
# Example usage configurations
# ========================================

# Configuration 1: Fixed learning rate (no changes)
"""
python main.py \
    --lr_scheduler_type constant \
    --learning_rate 1e-4 \
    --warmup_steps 0
"""

# Configuration 2: Start high, decay to 10% of initial, then stay constant
"""
python main.py \
    --lr_scheduler_type linear_decay_to_min \
    --learning_rate 1e-4 \
    --min_lr_ratio 0.1 \
    --warmup_steps 100 \
    --decay_start_step 500
"""

# Configuration 3: Cosine decay to minimum, then constant
"""
python main.py \
    --lr_scheduler_type cosine_decay_to_min \
    --learning_rate 1e-4 \
    --min_lr_ratio 0.05 \
    --warmup_steps 100 \
    --decay_start_step 1000
"""

# Configuration 4: Step decay (reduce by half every 1000 steps until minimum)
"""
python main.py \
    --lr_scheduler_type step_decay_to_min \
    --learning_rate 1e-4 \
    --min_lr_ratio 0.1 \
    --lr_step_size 1000 \
    --lr_decay_factor 0.5
"""

# Configuration 5: Fine-tuning with gentle decay
"""
python finetune.py \
    --lr_scheduler_type linear_decay_to_min \
    --learning_rate 1e-5 \
    --min_lr_ratio 0.2 \
    --warmup_steps 50 \
    --decay_start_step 1000
"""