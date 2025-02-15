#!/bin/bash

# MERIT Quick Start Script
# Usage: ./quick_start.sh

python main.py \
    --dataroot YOUR_DATA_PATH \          
    --num_classes CLASSES \
    --num_local_views NUM_LOCAL_VIEWS \
    --task TASK_ID \                     
    --batch_size BATCH_SIZE \            
    --num_workers NUM_WORKERS \          
    --window_size WINDOW_SIZE \          
    --patch_size PATCH_SIZE \            
    --step_size STEP_SIZE \              
    --lr LEARNING_RATE \                
    --epochs NUM_EPOCHS \                
    --coef LOSS_COEF \                   
    --device DEVICE_ID \                 
    --pretrain PRETRAIN_MODE1 PRETRAIN_MODE2  
    --glb USE_GLOBAL \                   
    \
    --output_marker RUN_ID \             
    --epochs_per_vali VAL_INTERVAL \     
    --cross_val USE_CROSSVAL \           
    --rand_seed RAND_SEED \              
    --train_prior USE_TRAIN_PRIOR \      
    --test_prior USE_TEST_PRIOR \        
    --test_cut CUT_RATIO \       
    --combine FUSION_METHOD              