kill -9 `ps -ef | grep -i "python backend/main.py" | grep -v grep | awk '{print $2}'` > /dev/null 2>&1

echo -n "Starting Kbot3.0 service (Estimated time cost: ~10 seconds)"

cd /home/ubuntu/kbot3

export KBOT_ENV=development
export PYTHONPATH=/home/ubuntu/kbot3

source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate kbot3

nohup python backend/main.py > /home/ubuntu/kbot3/service.log 2>&1 &

while true
do
        result=`tail /home/ubuntu/kbot3/service.log`
        if [[ $result == *"Application startup complete"* ]]; then
                echo 'Service started.'
                break
        else
                echo -n '.'
                sleep 1
        fi
done

cd - > /dev/null