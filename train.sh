# 실행: bash train.sh (permission denied의 경우 chmod -x train.sh)

# 이 리스트만 변경 - 실행할 config 이름 저장
CONFIGS=("base_config_pooling_mean" "base_config_pooling_max" "base_config_pooling_mean_max")

for (( i=0; i<${#CONFIGS[@]}; i++ ))
do
    python3 main.py -mt -c ${CONFIGS[$i]}
done