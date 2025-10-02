cuda_index=$1

# x_ref (ref_dir)  --perturb-->  x_adv (data_dir)  --generate-->  x_gen (gen_dir)
ref_dir=$2
data_dir=$3
gen_dir=$4

path_to_state_dict=./protector_model.pt
epsilon=0.07
clip_epsilon=1


cd "$(git rev-parse --show-toplevel)"

# Perturb

CUDA_VISIBLE_DEVICES=$cuda_index python -m scripts.evaluation.perturb_get_metric \
--in_channels 5 \
--epsilon ${epsilon} \
--clip_epsilon ${clip_epsilon} \
--clean_data_dir ${ref_dir} \
--save_dir ${data_dir}/IDProtector/clean \
--path_to_state_dict ${path_to_state_dict} \
--metrics_save_path ${data_dir}/IDProtector/clean/metrics.csv \
--batch_size 1 \
--cuda 0

read -p "Perturb complete. Press Enter to continue..."


# Distort

CUDA_VISIBLE_DEVICES=$cuda_index python -m scripts.evaluation.augmentation_dir_to_dir \
--source_dir ${data_dir}/IDProtector/clean \
--save_dir ${data_dir}/IDProtector

read -p "Distort complete. Press Enter to continue..."









# Generate

distortion_names=("compressed" "noisy" "cropped" "affine" "clean")
prompts=(
    "A portrait img of a person, coloured, realistic"
    "A photograph img of a person in detail"
    "A person img captured with realistic detail"
    "A close-up img of a person with lifelike features"
    "An img of a person, realistic, good quality, best quality"
)




cd "$(git rev-parse --show-toplevel)"



metric_paths=""

for distortion_name in "${distortion_names[@]}"
do
    for prompt_index in "${!prompts[@]}"; do

        CUDA_VISIBLE_DEVICES=$cuda_index python -m scripts.ip_inference.instantid \
        --source_dir ${data_dir}/IDProtector/${distortion_name} \
        --destination_dir ${gen_dir}/IDProtector/InstantID/${distortion_name}/prompt_${prompt_index} \
        --prompt "${prompts[$prompt_index]}"

        CUDA_VISIBLE_DEVICES=$cuda_index python -m scripts.evaluation.eval_ip_gen_quality \
        --generated_image_dir ${gen_dir}/IDProtector/InstantID/${distortion_name}/prompt_${prompt_index} \
        --ref_image_dir ${ref_dir} \
        --adv_image_dir ${data_dir}/IDProtector/${distortion_name} \
        --metric_savepath ${gen_dir}/IDProtector/InstantID/${distortion_name}/prompt_${prompt_index}/metric.csv

        metric_paths+="${gen_dir}/IDProtector/InstantID/${distortion_name}/prompt_${prompt_index}/metric.csv "

    done
done

python -m scripts.misc.merge_metrics \
--metric_paths ${metric_paths} \
--save_path ${gen_dir}/IDProtector/InstantID/metric.csv

python -m scripts.misc.merge_metrics \
--input_path ${gen_dir}/IDProtector/InstantID/metric.csv

read -p "InstantID complete. Press Enter to continue..."



metric_paths=""

for distortion_name in "${distortion_names[@]}"
do
    for prompt_index in "${!prompts[@]}"; do

        CUDA_VISIBLE_DEVICES=$cuda_index python -m scripts.ip_inference.ip_adapter \
        --model IPAdapter \
        --source_dir ${data_dir}/IDProtector/${distortion_name} \
        --destination_dir ${gen_dir}/IDProtector/IPAdapter/${distortion_name}/prompt_${prompt_index} \
        --prompt "${prompts[$prompt_index]}"

        CUDA_VISIBLE_DEVICES=$cuda_index python -m scripts.evaluation.eval_ip_gen_quality \
        --generated_image_dir ${gen_dir}/IDProtector/IPAdapter/${distortion_name}/prompt_${prompt_index} \
        --ref_image_dir ${ref_dir} \
        --adv_image_dir ${data_dir}/IDProtector/${distortion_name} \
        --metric_savepath ${gen_dir}/IDProtector/IPAdapter/${distortion_name}/prompt_${prompt_index}/metric.csv

        metric_paths+="${gen_dir}/IDProtector/IPAdapter/${distortion_name}/prompt_${prompt_index}/metric.csv "

    done
done

python -m scripts.misc.merge_metrics \
--metric_paths ${metric_paths} \
--save_path ${gen_dir}/IDProtector/IPAdapter/metric.csv

python -m scripts.misc.merge_metrics \
--input_path ${gen_dir}/IDProtector/IPAdapter/metric.csv

read -p "IPAdapter complete. Press Enter to continue..."




metric_paths=""

for distortion_name in "${distortion_names[@]}"
do
    for prompt_index in "${!prompts[@]}"; do

        CUDA_VISIBLE_DEVICES=$cuda_index python -m scripts.ip_inference.ip_adapter \
        --model IPAdapterPlus \
        --source_dir ${data_dir}/IDProtector/${distortion_name} \
        --destination_dir ${gen_dir}/IDProtector/IPAdapterPlus/${distortion_name}/prompt_${prompt_index} \
        --prompt "${prompts[$prompt_index]}"

        CUDA_VISIBLE_DEVICES=$cuda_index python -m scripts.evaluation.eval_ip_gen_quality \
        --generated_image_dir ${gen_dir}/IDProtector/IPAdapterPlus/${distortion_name}/prompt_${prompt_index} \
        --ref_image_dir ${ref_dir} \
        --adv_image_dir ${data_dir}/IDProtector/${distortion_name} \
        --metric_savepath ${gen_dir}/IDProtector/IPAdapterPlus/${distortion_name}/prompt_${prompt_index}/metric.csv

        metric_paths+="${gen_dir}/IDProtector/IPAdapterPlus/${distortion_name}/prompt_${prompt_index}/metric.csv "

    done
done

python -m scripts.misc.merge_metrics \
--metric_paths ${metric_paths} \
--save_path ${gen_dir}/IDProtector/IPAdapterPlus/metric.csv

python -m scripts.misc.merge_metrics \
--input_path ${gen_dir}/IDProtector/IPAdapterPlus/metric.csv

read -p "IPAdapterPlus complete. Press Enter to continue..."




metric_paths=""

for distortion_name in "${distortion_names[@]}"
do
    for prompt_index in "${!prompts[@]}"; do

        CUDA_VISIBLE_DEVICES=$cuda_index python -m scripts.ip_inference.photomaker \
        --source_dir ${data_dir}/IDProtector/${distortion_name} \
        --destination_dir ${gen_dir}/IDProtector/PhotoMaker/${distortion_name}/prompt_${prompt_index} \
        --prompt "${prompts[$prompt_index]}"

        CUDA_VISIBLE_DEVICES=$cuda_index python -m scripts.evaluation.eval_ip_gen_quality \
        --generated_image_dir ${gen_dir}/IDProtector/PhotoMaker/${distortion_name}/prompt_${prompt_index} \
        --ref_image_dir ${ref_dir} \
        --adv_image_dir ${data_dir}/IDProtector/${distortion_name} \
        --metric_savepath ${gen_dir}/IDProtector/PhotoMaker/${distortion_name}/prompt_${prompt_index}/metric.csv

        metric_paths+="${gen_dir}/IDProtector/PhotoMaker/${distortion_name}/prompt_${prompt_index}/metric.csv "

    done
done

python -m scripts.misc.merge_metrics \
--metric_paths ${metric_paths} \
--save_path ${gen_dir}/IDProtector/PhotoMaker/metric.csv

python -m scripts.misc.merge_metrics \
--input_path ${gen_dir}/IDProtector/PhotoMaker/metric.csv
