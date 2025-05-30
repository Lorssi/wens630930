#! /usr/bin/env python
# encoding:utf-8

"""
  类描述：  
  关于时间操作的工具类

  @version 1.0  
  
  Version    Date       ModifiedBy                 Content  
  -------- ---------    ----------         ------------------------  
  1.0      2017-7-11      zengzhj 
  1.1      2017-7-12      zengzhj            增加时间戳转换函数
  1.3 
"""

import time
import datetime
import time

version = 1.3


def time_stamp_long2short(time_stamp_long):
    """
    13位时间搓转为10+3位(浮点)形式
    """
    return int(time_stamp_long) * 1.0 / 1000


def time_stamp_sort2long(time_stamp_short):
    """
    10+3位(浮点)时间搓转为13位形式
    """
    return int(time_stamp_short) * 1000


def time2time_stamp_long(year, month, day, hour=0, minute=0, sec=0):
    """
    输入时间，返回mongo支持的NumberLong数据格式(13位整数)
    """
    time_stamp_long = time2time_stamp(year, month, day, hour=0, minute=0, sec=0) * 1000
    return time_stamp_long


def time2time_stamp(year, month, day, hour=0, minute=0, sec=0):
    """
    转换为10+3位(浮点)时间戳形式
    """
    year = str(year)
    month = str(month)
    day = str(day)
    hour = str(hour)
    minute = str(minute)
    sec = str(sec)

    time_str = year + month + day + hour + minute + sec
    time_tuple = time.strptime(time_str, "%Y%m%d%H%M%S")
    time_stamp = int(time.mktime(time_tuple))

    return time_stamp


def timestamp2datetime(timestamp, convert_to_local=False):
    ''' Converts UNIX timestamp(10位) to a datetime object. '''
    if isinstance(timestamp, (int, long, float)):
        dt = datetime.datetime.utcfromtimestamp(timestamp)
        if convert_to_local:  # 是否转化为本地时间
            dt = dt + datetime.timedelta(hours=8)  # 中国默认时区
        return dt
    return timestamp


def time_str2long_stamp(time_str, time_str_format="%Y%m%d:%H-%M-%S"):
    """
    """
    stamp = int(time.mktime(time.strptime(time_str, time_str_format)))
    long_stamp = time_stamp_sort2long(stamp)  # 目标日期的起始时刻

    return long_stamp


def func_time_cost(fn):
    def inner(*arg, **kwarg):
        s_time = time.time()
        res = fn(*arg, **kwarg)
        e_time = time.time()
        # print('{} 耗时：{}秒'.format(fn.__name__, e_time - s_time))
        return res

    return inner


if __name__ == "__main__":
    """
    测试
    """
    pass
