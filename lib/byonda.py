import numpy as np
from sklearn.datasets import make_moons, make_circles

class Life () :
    def __init__(self, name) :
        self.name = name
        
    def breath(self) :
        '''생명체들은 호흡을 한다.'''
        print("I'm alive.")
        
class Animal(Life) :
    def animate(self) :
        '''동물은 스스로 움직일 수 있다.'''
        print("I'm moving.")
    
class Vertebrate (Animal) :
    '''척추동물은 몸 안에 뼈를 가지고 있다.'''
    inner_bone = True
    
class Mammal (Vertebrate) :
    '''포유류는 비뇨기를 가지고 있다.'''
    urinary = True
        
class Predator (Mammal) :
    '''식육목은 열육치를 가지고 있다.'''
    carnassials = True
        
class Cat (Predator) :
    '''고양이는 자기 발바닥을 혀로 햝을 수 있다.'''
    def role_paws(self) :
        print("I'm taking a bath.")
        
class Ape(Mammal) :
    '''영장류는 엄지와 검지를 서로 만나게 할 수 있고, 주먹도 쥘 수 있다.'''
    def clench(self) :
        print("Take my Punch!")
        
class Human (Ape) :
    '''인간은 영장류 중 유일하게 직립보행을 할 수 있다.'''
    def erectus (self) :
        print("I'm walking !")
        
def get_contour_values(xx, yy) :
    return np.sin(xx) ** 10 + np.cos(10 + yy * xx) * np.cos(xx)

def get_linear_data(lenth) :
    rng = np.random.RandomState(42)
    
    x = 10 * rng.rand(lenth)
    y = 2 * x - 1 + rng.randn(lenth)
    
    return x, y

# 정규분포를 따르는 합성데이터 생성함수 정의
def generate_normal(n_samples, train_test_ratio=0.8, seed=2019):
    np.random.seed(seed)
    n = n_samples // 2
    n_train = int(n * train_test_ratio)
    X1 = np.random.normal(loc=10, scale=5, size=(n, 2))
    X2 = np.random.normal(loc=20, scale=5, size=(n, 2))
    Y1 = np.ones(n)
    #Y2 = - np.ones(n)
    Y2 = np.zeros(n)
    X_train = np.concatenate((X1[:n_train], X2[:n_train]))
    X_test = np.concatenate((X1[n_train:], X2[n_train:]))
    Y_train = np.concatenate((Y1[:n_train], Y2[:n_train]))
    Y_test = np.concatenate((Y1[n_train:], Y2[n_train:]))
    return (X_train, Y_train), (X_test, Y_test)