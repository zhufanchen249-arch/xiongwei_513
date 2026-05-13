source ~/miniconda3/etc/profile.d/conda.sh
source ./setup_so100_env.sh
python -m lerobot.teleoperate \
--robot.type=so100_follower \
--robot.port=/dev/ttyACM0 \
--teleop.type=keyboard