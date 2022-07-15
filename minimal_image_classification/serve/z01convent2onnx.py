"""
@Auth: itmorn
@Date: 2022/7/14-17:14
@Email: 12567148@qq.com
"""
from multiprocessing import Pool, cpu_count
import os
import traceback


def thread_task(url):
    try:
        os.system(
            "curl http://127.0.0.1:8080/predictions/densenet161 -T /data01/zhaoyichen/work_github/image_classifier/kitten.jpg")
        print(url)
    except Exception as e:
        print("Exception: " + str(e))
        traceback.print_exc()
        # raise Exception(str(e))


if __name__ == '__main__':

    lst_lines = [1, 2, 3]*100
    p = Pool(50)
    print("cpu数量为：%d" % cpu_count())
    print("主线程id为: %d" % os.getpid())
    print("线程开始处理了")
    try:
        # 线程池有3个线程, 线程数量可以大于cpu_count()的数量, 且os.getpid()获取的数值都不一样
        result_list = p.map(thread_task, lst_lines)
        print(result_list)

        print("等待所有线程执行完成")
        p.close()
        p.terminate()
    except Exception as e:
        print(e)
    finally:
        print("===============close===============")
        p.close()
        p.terminate()

