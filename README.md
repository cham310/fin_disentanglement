# Two-Step Disentanglement for Financial Data
This is Pytorch implementation of ["Two-Step Disentanglement for Financial Data"](https://arxiv.org/abs/1709.00199). 
![image](img/model.png)
Using this method, we disentangled topic-wise and market-wise impact on Korean stock prices. 
- ["Clustering National Security Threats Using Two-Step Disentanglement Method On Stock Prices"](https://lyusungwon.github.io/assets/publications/[2018%EC%B6%94%EA%B3%84]%EC%84%9C%EC%9A%B8%EB%8C%80%ED%95%99%EA%B5%90_%EC%82%B0%EC%97%85%EA%B3%B5%ED%95%99%EA%B3%BC_%EC%B5%9C%EB%AF%BC.pdf), Minh Choi, Sungwon Lyu, Sungzoon Cho, Korea Data Mining Society 2018 Fall Conference, Special Session Best Paper

# Requirements
- Pytorch 0.4

# Result
## Loss
Accuracy of the first step(left) and loss of the second step(right).
![image](img/result1.png)

## Embedding of Z
Embedded Zs of unstable periods(left) and of peaceful periods(right).
![image](img/result2.png)

## Analysis of Z 
The distribution of Zs of major provocations by North Korea. 
![image](img/result3.png)
