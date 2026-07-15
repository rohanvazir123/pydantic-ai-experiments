import threading

# define singleon
def singleton(cls):
    instances: dict = {}   
    lock = threading.Lock()
    
    def get_instance(*args, **kwargs) -> object:
        with lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
            return instances[cls]
        
    return get_instance
    

@singleton
class TestSingleton:
    def __init__(self,  *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        print(f"args = {args}")
        print(f"kwargs = {kwargs}")

# main
if __name__  == '__main__':
    print("hello")
    x1 = TestSingleton(1,2,3, key=3, val=4)
    x2 = TestSingleton(3,4,5, key=3, val=5)
    print(x1 == x2)
    print(x1.args, x1.kwargs)
    print(x2.args, x2.kwargs)
