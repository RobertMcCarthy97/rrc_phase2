# Real Robot Challenge Phase 2 attempt - ThriftySnipe

Our report from phase 2 can be found [here](https://openreview.net/forum?id=S6ckmbx9SiV).

The code is built off the [rrc_example_package](https://github.com/rr-learning/rrc_example_package/tree/master)
provided by the challenge organisers. For more details on how to use this code with Singularity and ROS 2, see
the relevant [documentation](https://people.tuebingen.mpg.de/felixwidmaier/rrc2021/singularity.html).

## Singularity Iimage

1. Download our custom singularity image: [user_image.sif](https://drive.google.com/drive/folders/1AKf4O28h8sYF_6J3FUq9oXJBY88joDcl?usp=sharing).
Otherwise, rebuild it yourself using 'user_image.def' and following
[these instructions](https://people.tuebingen.mpg.de/felixwidmaier/rrc2021/singularity.html#add-custom-dependencies-to-the-container).

2. Name the image `user_image.sif`.

## Train in Simulation

**WARNING:** Our method does not succesfully solve the Phase 2 rearrange dice task.

To train in simulation using domain randomization:

    singularity run /path/to/user_image.sif mpirun -np 8 python3 train.py --exp-dir='dr_training' --n-epochs=200

This will start training with just 1 dice in the arena, and will add 4 dice every 50 epochs untill there are 9 dice in total. Details of all relevant arguments are found in `rrc_example_package/her/arguments.py`.

### Real Robot

To deploy the trained model:

1. Find the trained model at `rrc_example_package/her/saved_models/dr_training/acmodel.pt`.

2. Upload the model to the robot cluster following [these instructions](https://people.tuebingen.mpg.de/felixwidmaier/rrc2021/submission_system/submission_system.html#upload-the-file).
The path to the model on the cluster should thus be `/userhome/acmodel.pt`. 

3. Ensure your [configuration file](https://people.tuebingen.mpg.de/felixwidmaier/rrc2021/submission_system/submission_system.html#configuration-file-roboch-json)
links to this repository.

4. [Login](https://people.tuebingen.mpg.de/felixwidmaier/rrc2021/submission_system/submission_system.html#submitting-a-job) via ssh and call `submit`.

This should run the `rrc_example_package/scripts/evaualte_stage2.py` script on the real robot.


<br/><br/>
# README from the original Example Package:

This is a basic example for a package that can be submitted to the robots of
the [Real Robot Challenge 2021](https://real-robot-challenge.com).

It is a normal ROS2 Python package that can be build with colcon.  However,
there are a few special files in the root directory that are needed for
running/evaluating your submissions.  See the sections on the different
challenge phases below for more on this.

This example uses purely Python, however, any package type that can be built
by colcon is okay.  So you can, for example, turn it into a CMake package if you
want to build C++ code.  For more information on this, see the [ROS2
documentation](https://docs.ros.org/en/foxy/Tutorials/Creating-Your-First-ROS2-Package.html).


Challenge Simulation Phase (Pre-Stage)
--------------------------------------

There are two example scripts using the simulation:

- `sim_move_up_and_down`:  Directly uses the `TriFingerPlatform` class to simply
  move the robot between two fixed positions.  This is implemented in
  `rrc_example_package/scripts/sim_move_up_and_down.py`.

- `sim_trajectory_example_with_gym`:  Wraps the robot class in a Gym environment
  and uses that to run a dummy policy which simply points with one finger on the
  goal positions of the trajectory.  This is implemented in 
  `rrc_example_package/scripts/sim_trajectory_example_with_gym.py`.

To execute the examples, [build the
package](https://people.tuebingen.mpg.de/felixwidmaier/rrc2021/singularity.html#singularity-build-ws)
and execute

    ros2 run rrc_example_package <example_name>


For evaluation of the pre-stage of the challenge, the critical file is the
`evaluate_policy.py` at the root directory of the package.  This is what is
going to be executed by `rrc_evaluate_prestage.py` (found in `scripts/`).

For more information, see the [challenge
documentation](https://people.tuebingen.mpg.de/felixwidmaier/rrc2021/)

`evaluate_policy.py` is only used for the simulation phase and not relevant
anymore for the later phases that use the real robot.


Challenge Real Robot Phases (Stages 1 and 2)
--------------------------------------------

For the challenge phases on the real robots, you need to provide the following
files at the root directory of the package such that your jobs can executed on
the robots:

- `run`:  Script that is executed when submitting the package to the robot.
  This can, for example, be a Python script or a symlink to a script somewhere
  else inside the repository.  In the given example, it is a shell script
  running a Python script via `ros2 run`.  This approach would also work for C++
  executables.  When executed, a JSON string encoding the goal is passed as
  argument (the exact structure of the goal depends on the current task).
- `goal.json`:  Optional.  May contain a fixed goal (might be useful for
  testing/training).  See the documentation of the challenge tasks for more
  details.

It is important that the `run` script is executable.  For this, you need to do
two things:

1. Add a shebang line at the top of the file (e.g. `#!/usr/bin/python3` when
   using Python or `#!/bin/bash` when using bash).
2. Mark the file as executable (e.g. with `chmod a+x run`).

When inside of `run` you want to call another script using `ros2 run` (as it is
done in this example), this other script needs to fulfil the same requirements.
