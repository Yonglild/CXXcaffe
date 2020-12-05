//
// Created by wyl on 2020/11/28.
//
#include<iostream>
#include<memory>

int main(){
    std::shared_ptr<int> p1(new int(1)); //方式1
    std::shared_ptr<int> p2 = p1; //方式2
    std::shared_ptr<int> p3;
//方式3 reset，如果原有的shared_ptr不为空，会使原对象的引用计数减1
    p3.reset(new int(1));
    //方式4
    std::shared_ptr<int> p4 = std::make_shared<int>(2);

//使用方法例子：可以当作一个指针使用
    std::cout << *p4 << std::endl;
    //std::shared_ptr<int> p4 = new int(1);
    if(p1) { //重载了bool操作符
        std::cout << "p is not null" << std::endl;
    }
    int* p = p1.get();//获取原始指针
    std::cout << *p << std::endl;
}