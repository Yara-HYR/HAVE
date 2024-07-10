#!/bin/bash





SUIT=carla
# DOMAIN=tunnel
DOMAIN=highway
#highspeed
WEATHER=dense_fog
SPEED=60
AGENT=deepmdp
PERCEPTION=RGB-frame
#PERCEPTION=DVS-frame
#PERCEPTION=DVS-stream
# PERCEPTION=RGB-frame+DVS-voxel-grid
#PERCEPTION=RGB-frame+DVS-frame

#ENCODER=pixelFusedMaskV2
ENCODER=pixelCarla098
DECODER=identity

RPC_PORT=7632
TM_PORT=17632

RPC_PORT_EVAL=4152
TM_PORT_EVAL=14152

CUDA_DEVICE=1

SEED=111

UNIQUE_ID=${SUIT}+${DOMAIN}+${WEATHER}+${SPEED}+${AGENT}+${PERCEPTION}+${ENCODER}+${DECODER}+${SEED}
LOGFILE=./logs/${UNIQUE_ID}.log
WORKDIR=./logs/${UNIQUE_ID}


echo ${UNIQUE_ID}
mkdir -p ${WORKDIR}


CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python -u train_testm.py \
	--work_dir ${WORKDIR}$ \
	--suit carla \
	--domain_name ${DOMAIN} \
	--selected_weather ${WEATHER} \
	--agent ${AGENT} \
	--perception_type ${PERCEPTION} \
	--encoder_type ${ENCODER} \
	--decoder_type ${DECODER} \
	--action_model_update_freq 1 \
	--transition_reward_model_update_freq 1 \
	--carla_rpc_port ${RPC_PORT} \
	--carla_tm_port ${TM_PORT} \
	--carla_rpc_port_eval ${RPC_PORT_EVAL} \
	--carla_tm_port_eval ${TM_PORT_EVAL} \
    --carla_timeout 30 \
    --frame_skip 1 \
    --init_steps 1000 \
    --max_episode_steps 300 \
    --rl_image_size 128 \
    --num_cameras 1 \
    --actor_lr 1e-3 \
    --critic_lr 1e-3 \
    --encoder_lr 1e-3 \
    --decoder_lr 1e-3 \
    --replay_buffer_capacity 10000 \
    --batch_size 128 \
    --EVAL_FREQ_EPISODE 20 \
    --EVAL_FREQ_STEP 50000 \
    --num_eval_episodes 10 \
    --save_tb \
    --do_metrics \
    --seed ${SEED}>${LOGFILE} 2>&1 &

	
