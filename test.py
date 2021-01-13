# from multiprocessing import Pool, cpu_count
# import os
# import time


# def long_time_task(i):
#     print('子进程: {} - 任务{}'.format(os.getpid(), i))
#     time.sleep(2)
#     print("结果: {}".format(8 ** 20))


# if __name__=='__main__':
#     print("CPU内核数:{}".format(cpu_count()))
#     print('当前母进程: {}'.format(os.getpid()))
#     start = time.time()
#     p = Pool(cpu_count())
#     for i in range(cpu_count()):
#         p.apply_async(long_time_task, args=(i,))
#     print('等待所有子进程完成。')
#     p.close()
#     p.join()
#     end = time.time()
#     print("总共用时{}秒".format((end - start)))



# import time
# import multiprocessing
# def is_prime(n):
#       if (n <= 1) : 
#           return 'not a prime number'
#       if (n <= 3) : 
#           return 'prime number'
          
#       if (n % 2 == 0 or n % 3 == 0) : 
#           return 'not a prime number'
    
#       i = 5
#       while(i * i <= n) : 
#           if (n % i == 0 or n % (i + 2) == 0) : 
#               return 'not a prime number'
#           i = i + 6
    
#       return 'prime number'
# def multiprocessing_func(x):
#     time.sleep(2)
#     print('{} is {} number'.format(x, is_prime(x)))
    
# if __name__ == '__main__':
#     starttime = time.time()
#     processes = []
#     for i in range(1,10):
#         p = multiprocessing.Process(target=multiprocessing_func, args=(i,))
#         processes.append(p)
#         p.start()
        
#     for process in processes:
#         process.join()
        
#     print()    
#     print('Time taken = {} seconds'.format(time.time() - starttime))
pursuer = -1
danger_num = 1
if pursuer == -1 and (danger_num==1 or danger_num==2):
    print('yes')
else:
    print('no')