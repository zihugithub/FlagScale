import warnings

import hydra

from omegaconf import DictConfig, OmegaConf

from flagscale.runner.auto_tuner import AutoTuner, ServeAutoTunner
from flagscale.runner.runner_compress import SSHCompressRunner
from flagscale.runner.runner_inference import SSHInferenceRunner
from flagscale.runner.runner_rl import SSHRLRunner
from flagscale.runner.runner_serve import CloudServeRunner, SSHServeRunner
from flagscale.runner.runner_train import CloudTrainRunner, SSHTrainRunner
from flagscale.runner.utils import is_master

# To accommodate the scenario where the before_start field is used to switch to the actual environment during program execution,
# we have placed the import statements inside the function body rather than at the beginning of the file.


def check_and_reset_deploy_config(config: DictConfig) -> None:
    if config.experiment.get("deploy", {}):
        OmegaConf.set_struct(config.experiment.runner, False)
        config.experiment.runner.deploy = config.experiment.deploy
        del config.experiment.deploy
        warnings.warn(
            "'config.experiment.deploy' has been moved to 'config.experiment.runner.deploy'. "
            "Support for the old location will be removed in a future release."
        )
        OmegaConf.set_struct(config.experiment.runner, True)


@hydra.main(version_base=None, config_name="config")
def main(config: DictConfig) -> None:
    print("00000000000000000000000")
    print(config)
    print("1111111111111111111")
    check_and_reset_deploy_config(config)
    print(config)
    task_type = config.experiment.task.get("type", "train")
    print("222222222222222222222222")
    print(task_type)
    if task_type == "train":
        if config.action == "auto_tune":
            # For MPIRUN scene, just one autotuner process.
            # NOTE: This is a temporary solution and will be updated with cloud runner.
            if is_master(config):
                tuner = AutoTuner(config)
                tuner.tune()
        else:
            if config.experiment.runner.get("type", "ssh") == "ssh":
                runner = SSHTrainRunner(config)
            elif config.experiment.runner.get("type") == "cloud":
                runner = CloudTrainRunner(config)
            else:
                raise ValueError(f"Unknown runner type {config.runner.type}")

            if config.action == "run":
                runner.run()
            elif config.action == "dryrun":
                runner.run(dryrun=True)
            elif config.action == "test":
                runner.run(with_test=True)
            elif config.action == "stop":
                runner.stop()
            elif config.action == "query":
                runner.query()
            else:
                raise ValueError(f"Unknown action {config.action}")
    elif task_type == "inference":
        print("33333333333333333333")
        # runner = SSHInferenceRunner(config)
        # if config.action == "run":
        #     runner.run()
        # elif config.action == "dryrun":
        #     runner.run(dryrun=True)
        # elif config.action == "test":
        #     runner.run(with_test=True)
        # elif config.action == "stop":
        #     runner.stop()
        # else:
        #     raise ValueError(f"Unknown action {config.action}")
    elif task_type == "serve":
        if config.action == "auto_tune":
            # For MPIRUN scene, just one autotuner process.
            # NOTE: This is a temporary solution and will be updated with cloud runner.
            tuner = ServeAutoTunner(config)
            tuner.tune()
        else:
            if config.experiment.runner.get("type", "ssh") == "ssh":
                runner = SSHServeRunner(config)
            elif config.experiment.runner.get("type", "ssh") == "cloud":
                runner = CloudServeRunner(config)
            else:
                raise ValueError(f"Unknown runner type {config.runner.type}")
            if config.action == "run":
                runner.run()
            elif config.action == "test":
                runner.run(with_test=True)
            elif config.action == "stop":
                runner.stop()
            else:
                raise ValueError(f"Unknown action {config.action}")
    elif task_type == "compress":
        runner = SSHCompressRunner(config)
        if config.action == "run":
            runner.run()
        elif config.action == "dryrun":
            runner.run(dryrun=True)
        elif config.action == "stop":
            runner.stop()
        else:
            raise ValueError(f"Unknown action {config.action}")
    elif task_type == "rl":
        runner = SSHRLRunner(config)
        if config.action == "run":
            runner.run()
        elif config.action == "dryrun":
            runner.run(dryrun=True)
        elif config.action == "test":
            runner.run(with_test=True)
        elif config.action == "stop":
            runner.stop()
        else:
            raise ValueError(f"Unknown action {config.action}")
    else:
        raise ValueError(f"Unknown task type {task_type}")


if __name__ == "__main__":
    main()
