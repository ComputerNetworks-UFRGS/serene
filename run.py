import pathlib
import json
import pexpect
import sys
import argparse
import yaml
import math
import subprocess
import os
import time
import datetime

# Added './client/' to path to enable importing these files from outside their original module
sys.path.append("./client/")
from models import get_training_settings
from ml import num_params

PROJECT_ROOT = pathlib.Path(__file__).parent

auto_generated_message = "This file is generated automatically by the run.py script"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--stale", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--straggling-pattern", type=str, required=False)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--file", type=str, metavar="FILE")
    parser.add_argument("--table-cols", type=int, default=32)
    parser.add_argument("--table-rows", required=False, default="auto")
    parser.add_argument("--just-read-pkt", action="store_true")
    return parser.parse_args()


def clean():
    subprocess.run(["make", "-C", "p4src", "clean"])


def generate_cli_config(num_workers):
    cli_commands = []
    for worker in range(num_workers):
        cli_commands.append(f"mirroring_add {worker} {worker + 1}\n")

    cli_commands.append("set_queue_depth 10000\n")
    #cli_commands.append("set_queue_rate 3000\n")

    with open(PROJECT_ROOT / "p4src/switches/s1-cli.txt", "w") as cli_file:
        cli_file.writelines(cli_commands)


def generate_topology(num_workers):
    def worker_config(n):
        return {
            "ip": f"10.0.1.{n}/24",
            "mac": f"08:00:00:00:01:{n:02}",
            "commands": [
                f"route add default gw 10.0.1.{n}0 dev eth0",
                f"arp -i eth0 -s 10.0.1.10 08:00:00:00:{n:02}:00",
            ],
        }

    topology = {
        "hosts": {f"h{n}": worker_config(n) for n in range(1, num_workers + 1)},
        "switches": {
            "s1": {
                "runtime_json": "switches/s1-runtime.json",
                "cli_input": "switches/s1-cli.txt",
            }
        },
        "links": [[f"h{n}", f"s1-p{n}", "0", 100] for n in range(1, num_workers + 1)],
    }

    topology_path = PROJECT_ROOT / "p4src/topology.json"

    with open(topology_path, "w") as topology_file:
        json.dump(topology, topology_file, indent=4)


def generate_switch_rules(stale, num_workers):
    switch_config = {
        "target": "bmv2",
        "p4info": "build/stragglers.p4.p4info.txt",
        "bmv2_json": "build/stragglers.json",
        "table_entries": [
            {
                "table": "MyIngress.window_table",
                "match": {},
                "action_name": "MyIngress.set_window",
                "action_params": {"currentWindow": stale},
            }
        ],
    }

    for worker in range(num_workers):
        worker_number = worker + 1
        worker_entry = {
            "table": "MyEgress.worker_data",
            "match": {"standard_metadata.egress_port": worker_number},
            "action_name": "MyEgress.set_worker_data",
            "action_params": {
                "macAddr": f"08:00:00:00:01:{worker_number:02}",
                "ipAddr": f"10.0.1.{worker_number}",
                "workerId": worker,
            },
        }
        switch_config["table_entries"].append(worker_entry)

    switch_runtime = PROJECT_ROOT / "p4src/switches/s1-runtime.json"
    with open(switch_runtime, "w") as switch_config_file:
        json.dump(switch_config, switch_config_file, indent=4)


def generate_p4_constants(num_workers, model_size, window_size, columns, rows="auto"):
    ideal_window_size = 1
    while ideal_window_size < (window_size + 1):
        ideal_window_size *= 2

    if rows == "auto":
        rows = math.ceil(model_size / columns)
    elif rows == "keep":
        rows = math.ceil(model_size / 32)
    else:
        rows = int(rows)

    consts = {
        "MAX_WORKERS": num_workers,
        "BUFFER_SIZE": ideal_window_size,
        "AGGR_ROWS": rows,
        "VALUES_PER_ROW": columns,
    }
    table_size = rows * columns
    used_bytes = table_size * 4
    buffer_size = ideal_window_size * num_workers + 32

    with open("mem_report.txt", "w") as mem_report:
        print(f"Total table size: {rows} x {columns} = {table_size}", file=mem_report)
        print(
            f"Table memory usage: {used_bytes} bytes = {used_bytes/1000000} MB",
            file=mem_report,
        )
        print(
            f"Buffer size: {ideal_window_size} x {num_workers} + 32 = {buffer_size} bits",
            file=mem_report,
        )

    with open(PROJECT_ROOT / "p4src/includes/constants.p4", "w") as f:
        f.write(f"// {auto_generated_message}\n")
        for name, val in consts.items():
            f.write(f"#define {name} {val}\n")


def generate_p4_actions(columns, just_read):
    with open(PROJECT_ROOT / "p4src/includes/ssp.p4", "w") as ssp_p4:
        ssp_p4.write(f"// {auto_generated_message}\n")
        ssp_p4.write(
            "action aggregate(bit<16> baseIndex, bit<32> element, bit<32> value) {\n"
            "    bit<32> currentValue;\n"
            "    bit<32> index = (bit<32>) baseIndex * VALUES_PER_ROW + element;\n"
            "    sspTable.read(currentValue, index);\n"
        )

        if not just_read:
            ssp_p4.write("    sspTable.write(index, currentValue + value);\n")
        ssp_p4.write("}\n\n")

        ssp_p4.write(
            "action readRow() {\n"
            f"    hdr.ipv4.totalLen = hdr.ipv4.totalLen + {columns * 4};\n"
            f"    hdr.udp.length = hdr.udp.length + {columns * 4};\n"
            "    hdr.data.setValid();\n"
            "    bit<32> index = (bit<32>) hdr.ssp.gradSegment * VALUES_PER_ROW;\n"
        )

        for i in range(columns):
            ssp_p4.write(f"    sspTable.read(hdr.data.value{i:02}, index + {i});\n")
        ssp_p4.write("}\n\n")
        ssp_p4.write("action increment() {\n")
        for i in range(columns):
            ssp_p4.write(
                f"    aggregate(hdr.ssp.gradSegment, {i}, hdr.data.value{i:02});\n"
            )
        ssp_p4.write("}\n\n")


def generate_py_constants(columns, rows):
    if rows == "keep":
        columns = 32
    with open(PROJECT_ROOT / "client/constants.py", "w") as py_consts:
        py_consts.write(f"# {auto_generated_message}\n")
        py_consts.write(
            f"GRADS_PER_PKT = {columns}\n"
            f"SCALING_FACTOR = 10**8\n"
            f"MIN_INT = -(2**32)\n"
            f"MAX_INT = 2**32 - 1\n"
        )


def run_experiment(mininet_proc, num_workers, model, straggling_pattern, name="./"):
    base_cmd = (
        "h{} python3 ../client/worker.py {} --model {} --world_size {} --out-dir {}"
    )

    for worker in range(num_workers):
        worker_cmd = base_cmd
        if straggling_pattern["workers"] == "all" or worker in straggling_pattern["workers"]:
            worker_cmd += " straggler "
            for key, value in straggling_pattern["args"].items():
                worker_cmd += f'--{key} {value} '
        worker_cmd += "&"
        mininet_proc.sendline(
            worker_cmd.format(worker + 1, worker, model, num_workers, name)
        )
        mininet_proc.expect("mininet> ")

def count_finished(output_dir):
    finish_files = [file for file in output_dir.iterdir() if ".finish" in file.name]
    return len(finish_files)

def main():
    args = get_args()

    assert (args.file is not None) or (
        args.workers is not None and args.stale is not None
    )

    if args.straggling_pattern:
        pattern = json.loads(
            pathlib.Path(args.straggling_pattern).read_text()
        )
    else:
        pattern = {}
    if args.file is not None:
        with open(args.file, "r") as f:
            data = yaml.load(f, yaml.Loader)
    else:
        now = datetime.datetime.now()
        experiment_name = now.strftime("%Y%m%d-%H%M")
        data = {
            "experiments": {
                experiment_name: {
                    "model": args.model,
                    "workers": args.workers,
                    "stale": args.stale,
                    "cols": args.table_cols,
                    "rows": args.table_rows,
                    "just_read": args.just_read_pkt,
                    "straggling_pattern": pattern
                }
            }
        }

    for exp, parameters in data["experiments"].items():
        clean()

        training_settings = get_training_settings(parameters["model"])
        model_size = num_params(training_settings.model)

        generate_cli_config(parameters["workers"])
        generate_topology(parameters["workers"])
        generate_switch_rules(parameters["stale"], parameters["workers"])
        generate_p4_constants(
            parameters["workers"],
            model_size,
            parameters["stale"],
            parameters["cols"],
            parameters["rows"],
        )
        generate_p4_actions(parameters["cols"], parameters["just_read"])
        generate_py_constants(parameters["cols"], parameters["rows"])

        mininet_proc = pexpect.spawn("make -C p4src", encoding="utf-8")

        if args.interactive:
            mininet_proc.interact()
        else:
            mininet_proc.logfile = sys.stdout
            mininet_proc.expect("mininet> ")
            output_dir = PROJECT_ROOT / "results" / exp
            output_dir.mkdir(parents=True, exist_ok=True)

            settings_file = output_dir / "settings.json"
            settings_file.touch(exist_ok=True)
            settings_file.write_text(json.dumps(parameters, indent=4))

            run_experiment(
                mininet_proc, parameters["workers"], parameters["model"], parameters["straggling_pattern"], output_dir,
            )

            while count_finished(output_dir) != parameters["workers"]:
                time.sleep(5)

            mininet_proc.sendline("exit")




if __name__ == "__main__":
    main()
