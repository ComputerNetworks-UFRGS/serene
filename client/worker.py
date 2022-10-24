from pathlib import Path
from statistics import mean
from itertools import cycle
import argparse
import copy
import random
import sys
import time

import pandas as pd
import torch

from models import TrainingSettings, get_training_settings, get_models
from ssp import inc_rows, clock, read_rows
from ml import (
    accuracy,
    display_eval,
    evaluate,
    loss_batch,
    rebuild_format,
    flatten_and_split_tensors,
    apply_gradients,
    get_parameter_shapes,
)
from straggling_generator import StragglerGenerator

FINISH_FILE = Path.cwd() / "finished.txt"

class Worker:
    def __init__(
        self,
        id,
        settings: TrainingSettings,
        clock=0,
        world_size=8,
        out_dir=Path(),
        straggler_generator=None
    ):
        self.id = id
        self.clock = clock
        self.straggler_generator=straggler_generator

        self.model = settings.model
        self.steps = settings.steps // world_size
        self.lr = settings.lr
        self.loss_fn = settings.loss
        self.optim_fn = settings.optimizer
        self.train_dl = settings.train_dl
        self.test_dl = settings.test_dl
        self.target_accuracy = settings.target_accuracy

        self.world_size = world_size

        self.out_file = out_dir / f"worker-{self.id}-stats.csv"
        self.out_model = out_dir / f"worker-{self.id}-model.pth"
        self.final_file = out_dir / f"worker-{self.id}.finish"

        self.total_time = 0
        self.waiting_time = 0
        self.iter_times = []

        self.val_losses = []
        self.val_accs = []
        self.train_losses = []
        self.train_accs = []
        self.time_refs = []

    def run(self):
        print(f"Will straggle using: {self.straggler_generator}")
        initial_time = time.time()
        # Calculating the parameter sizes
        parameter_shapes = get_parameter_shapes(self.model)
        splitted_tensors = flatten_and_split_tensors(self.model.parameters())
        num_splits = len(splitted_tensors)
        total_size = sum(shape.numel() for shape in parameter_shapes)

        loss, _, acc = evaluate(self.model, self.loss_fn, self.test_dl, accuracy)

        self.time_refs.append(time.time())
        self.val_accs.append(acc)
        self.val_losses.append(loss)

        print("Joining training... ")
        clock(self.id, 0)
        print("Joined!")

        iter_train = cycle(self.train_dl)
        sleep_time = 0

        for step in range(1, self.steps + 1):
            if FINISH_FILE.exists():
                print("Finishing by finish signal")
                break

            print(f"========== Step {step} ==========")

            iter_start = time.time()
            print("Getting new model...")
            current_model = self.get_new_model(
                step, num_splits, parameter_shapes, total_size
            )
            print("Checking straggling...")
            if self.iter_times and self.straggler_generator is not None:
                sleep_time = self.straggler_generator.step(mean(self.iter_times))

            print("Training...")
            
            current_model, acc = self.run_training_step(step, current_model, iter_train)

            print("Sending gradients...")
            
            self.send_gradients(step, current_model, num_splits)

            iter_end = time.time()
            self.iter_times.append(iter_end - iter_start - sleep_time)

            if acc > self.target_accuracy:
                FINISH_FILE.touch()

            print("Waiting clock...")
            self.signal_clock(step)

            if acc > self.target_accuracy:
                print("Finishing by hitting target accuracy")
                break

        self.total_time = time.time() - initial_time

        pd.DataFrame(
            {
                "train_loss": [None] + self.train_losses,
                "train_acc": [None] + self.train_accs,
                "val_loss": self.val_losses,
                "val_acc": self.val_accs,
                "time_refs": self.time_refs,
            }
        ).to_csv(self.out_file)
        torch.save(current_model, self.out_model)

        with open(self.out_file, "a") as f:
            f.write(f"{self.total_time},{self.waiting_time}\n")
        self.final_file.touch()
        

    def get_new_model(self, step, num_splits, parameter_shapes, total_size):
        self.model.zero_grad()
        current_model = copy.deepcopy(self.model)
        waiting_start = time.time()
        flat_grad = read_rows(num_splits, self.id, step)[:total_size]
        waiting_end = time.time()

        self.waiting_time += waiting_end - waiting_start

        grads = rebuild_format(flat_grad, parameter_shapes)
        current_model = apply_gradients(current_model, grads)

        return current_model

    def run_training_step(self, step, current_model, iter_train):
        current_model.train()
        images, labels = next(iter_train)

        opt = self.optim_fn(current_model.parameters(), lr=self.lr)
        train_loss, _, train_acc = loss_batch(
            current_model, self.loss_fn, images, labels, opt, accuracy
        )

        val_acc = 0
        if step % 10 == 0:
            train_loss, _, train_acc = loss_batch(
                current_model, self.loss_fn, images, labels, metric=accuracy
            )
            val_loss, _, val_acc = evaluate(
                current_model, self.loss_fn, self.test_dl, accuracy
            )

            display_eval(step, self.steps, train_loss, train_acc, val_loss, val_acc)
            self.time_refs.append(time.time())
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

        return current_model, val_acc

    def send_gradients(self, step, current_model, num_splits):
        with torch.no_grad():
            new_grads = flatten_and_split_tensors(
                [
                    tensor.grad * self.lr * (1 / self.world_size)
                    for tensor in current_model.parameters()
                ]
            )

            inc_rows(num_splits, new_grads, self.id, step)

    def signal_clock(self, step):
        clock_wait_start = time.time()
        clock(self.id, step)
        clock_wait_end = time.time()
        self.waiting_time += clock_wait_end - clock_wait_start


def get_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("worker_id", type=int)

    available_models = get_models()

    parser.add_argument(
        "--model", choices=available_models, default=available_models[0]
    )
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--out-dir", type=str, required=False)
    subparsers = parser.add_subparsers(dest="behavior")
    straggling_subparser = subparsers.add_parser("straggler")
    straggling_subparser.add_argument("--pattern", type=str, default="slow_worker")
    straggling_subparser.add_argument("--probability", type=float, required=False)
    straggling_subparser.add_argument("--min-slowdown", type=float, required=False)
    straggling_subparser.add_argument("--max-slowdown", type=float, required=False)
    straggling_subparser.add_argument("--failure-duration", type=float, required=False)
    straggling_subparser.add_argument("--constant-slowdown", type=float, required=False)

    return parser.parse_args()


def main():
    FINISH_FILE.unlink(missing_ok=True)

    args = get_args()
    if args.behavior == "straggler":
        # Passing all arguments required for every available pattern
        # Probably can be improved
        straggler_generator = StragglerGenerator.getInstance(
            args.pattern,
            probability=args.probability,
            min_slowdown=args.min_slowdown,
            max_slowdown=args.max_slowdown,
            failure_duration=args.failure_duration,
            constant_slowdown=args.constant_slowdown,
        )
    else:
        straggler_generator = None

    out_dir = Path(__file__).parent.parent / args.out_dir
    with open(out_dir / f"h{args.worker_id}.log", "w", buffering=1) as out:
        sys.stdout = out
        sys.stderr = out
        print(f"Training with model: {args.model}")
        sys.stdout.flush()
        settings = get_training_settings(args.model)

        worker = Worker(
            args.worker_id,
            settings,
            world_size=args.world_size,
            out_dir=out_dir,
            straggler_generator=straggler_generator,
        )

        initial_time = time.time()
        worker.run()
        end_time = time.time()
        print("Total time:", end_time - initial_time)


if __name__ == "__main__":
    main()
