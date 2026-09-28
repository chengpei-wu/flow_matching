```bash
python ./src/main.py --dataset single_mnist-8 --epochs 300 --batch_size 4096 --lr 0.001

# -----------
python ./src/main.py --dataset single_mnist-8 --sample
```

## Fixed-step trajectory supervision

Run these commands from the repository root. The existing
`single_mnist-8_flow_matching.pth` is used as a pretrained teacher. Each noise
sample is paired with the teacher's output from that **same** noise sample;
targets at step `k` are `(1-k/20)*noise + (k/20)*teacher_output`.

```bash
# 20 separate lightweight blocks: predict velocity, update x + v/20,
# minimize the mean of the 20 rollout-state reconstruction losses.
python -m src.stepwise --mode velocity_rec --dataset single_mnist-8 --steps 20

# 20 separate lightweight blocks: predict the entire next state directly,
# minimize exactly the same rollout-state reconstruction losses.
python -m src.stepwise --mode direct --dataset single_mnist-8 --steps 20

# Generate and save six frames, from noise to the final sample.
python -m src.stepwise --mode velocity_rec --dataset single_mnist-8 --steps 20 --sample
python -m src.stepwise --mode direct --dataset single_mnist-8 --steps 20 --sample
```

The methods write separate checkpoints (`*_velocity_rec_20step.pth` and
`*_direct_20step.pth`) and sample images (`*_trajectory.png`). Use
`--checkpoint PATH`, `--output PATH`, `--width`, `--batch-size`, `--epochs`,
`--teacher-checkpoint PATH`, and `--teacher-steps` to change these settings.
The same step count, width, mode, and image shape must be used when sampling.
For large images, reduce `--batch-size`: backpropagation retains all 20 blocks.

To reproduce the original independent Gaussian/image pairing instead, pass
`--coupling independent`. With rollout reconstruction, independent targets are
unpredictable from the sampled noise. Minimizing their MSE can drive the model
toward the dataset mean; it is included only as a controlled comparison. The
default teacher coupling is a **distillation** experiment: its final target
distribution is the teacher's, and image quality is bounded by that teacher.
Even a deterministic endpoint pairing does not guarantee that every linear
interpolation step is an injective map. Neither variant is a continuous-time
FM velocity estimator; sampling uses the trained number of blocks.

Run the tests in an environment with PyTorch, torchvision, matplotlib and tqdm:

```bash
python -m unittest discover -s tests
```
