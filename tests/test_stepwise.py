import unittest

import torch
from torch import nn

from src.stepwise import StepwiseTransport, teacher_endpoint


class ConstantBlock(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, x):
        return torch.ones_like(x) * self.value


class ConstantTeacher(nn.Module):
    def forward(self, x, t):
        return torch.ones_like(x) * 2


class StepwiseTests(unittest.TestCase):
    def test_velocity_rec_rollout_and_all_step_gradients(self):
        model = StepwiseTransport(1, steps=2, width=4, mode="velocity_rec")
        model.blocks = nn.ModuleList([ConstantBlock(2), ConstantBlock(0)])
        noise = torch.zeros(1, 1, 4, 4)
        target = torch.full_like(noise, 2)
        final, loss, trajectory = model(noise, target, return_trajectory=True)
        torch.testing.assert_close(trajectory[1], torch.ones_like(noise))
        torch.testing.assert_close(final, torch.ones_like(noise))
        torch.testing.assert_close(loss, torch.tensor(0.5))
        loss.backward()
        self.assertIsNotNone(model.blocks[0].value.grad)
        self.assertIsNotNone(model.blocks[1].value.grad)
        self.assertNotEqual(model.blocks[0].value.grad.item(), 0)
        self.assertNotEqual(model.blocks[1].value.grad.item(), 0)

    def test_direct_predicts_full_state_without_input_skip(self):
        model = StepwiseTransport(1, steps=2, width=4, mode="direct")
        model.blocks = nn.ModuleList([ConstantBlock(1), ConstantBlock(2)])
        noise = torch.full((1, 1, 4, 4), 7.0)
        result, _, trajectory = model(noise, return_trajectory=True)
        torch.testing.assert_close(trajectory[1], torch.ones_like(noise))
        torch.testing.assert_close(result, torch.full_like(noise, 2))

    def test_teacher_endpoint_pairs_each_noise_with_its_own_output(self):
        noise = torch.randn(2, 1, 4, 4)
        torch.testing.assert_close(
            teacher_endpoint(ConstantTeacher(), noise, steps=4), noise + 2
        )


if __name__ == "__main__":
    unittest.main()
