# 실행: bash train.sh (permission denied의 경우 chmod -x train.sh)

# 이 리스트만 변경 - 실행할 config 이름 저장
CONFIGS=("bin_loss_1" "bin_loss_3" "bin_loss_4" "bin_loss_5")

for (( i=0; i<${#CONFIGS[@]}; i++ ))
do
    python3 main.py -mt -c ${CONFIGS[$i]}
done