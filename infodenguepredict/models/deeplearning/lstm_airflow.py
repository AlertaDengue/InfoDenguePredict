u"""
Airflow workflow to train the LSTM model for multiple cities
"""

from builtins import range
import airflow
from airflow.operators.python_operator import PythonOperator
from airflow.models import DAG
from . import lstm

import time
from pprint import pprint

args = {
    'owner': 'airflow',
    'start_date': airflow.utils.dates.days_ago(2)
}

cities = [('RJ', 3304557), ('RJ', 3303500), ('RJ', 3301009)]
dag = DAG(
    dag_id='batch_train_models', default_args=args,
    schedule_interval=None)


def model_training(random_base):
    """This is a function that will run within the DAG execution"""
    time.sleep(random_base)


def print_context(ds, **kwargs):
    pprint(kwargs)
    print(ds)
    return 'Whatever you return gets printed in the logs'

run_this = PythonOperator(
    task_id='print_the_context',
    provide_context=True,
    python_callable=print_context,
    dag=dag)

# Generate 10 sleeping tasks, sleeping from 0 to 9 seconds respectively
for i in cities:
    task = PythonOperator(
        task_id='sleep_for_' + str(i),
        python_callable=my_sleeping_function,
        op_kwargs={'random_base': float(i) / 10},
        dag=dag)

    task.set_upstream(run_this)
