# for num in 3 4 5; do
#     python train_sa.py \
#     --project "MetaRace-SA" \
#     --name "ppo" \
#     --seed ${num} \
#     --mini_batch_size 600 \
#     --num_processes 12 \
#     --num_agents 1 
# done

# for num in 1 2 3; do
#     python train_sa.py \
#     --project "MetaRace-SA" \
#     --name "se-ppo" \
#     --use_representation_learning \
#     --seed ${num} \
#     --mini_batch_size 600 \
#     --num_processes 12 \
#     --num_agents 1 
# done

python train_sa.py \
    --project "MetaRace-SA-debug" \
    --name "ppo-normal" \
    --sample_dist "normal" \
    --seed 0 \
    --mini_batch_size 600 \
    --num_processes 12 \
    --num_agents 1 
