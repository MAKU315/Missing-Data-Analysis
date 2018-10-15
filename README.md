# Missing-Data-Analysis

## 결측치가 있는 교통데이터 역추정

결측치가 많은 교통 데이터의 결측치를 역추정하여 보정한다.

주어진 데이터의 양이 많지 않고 범주형 데이터로서 딥러닝 방법론 적용이 쉽지 않았다.

때문에, 머신러닝 방법론과 연계하여 분석을 진행하였다.

### 1. 머신러닝을 통한 케이스 분류
자주 발생하는 교통사고와 매우 간간히 발생하는 케이스로 나누어져 있었다.

때문에, 특별 케이스와 일반적인 케이스로 관측치를 구분하였다.

특별 케이스는 분포기반 방법론(연관성 분석)과 거리기반 방법론(KNN 등)을 바탕으로 결측치를 추정했으며,

일반적인 케이스는 Random Forest, Extra-Trees, SVM, Naive bayes 등의 머신러닝 기법들을 활용하여 결측치를 추정했다.

### 2. 딥러닝을 위한 재 보정

3~4개의 결측치 중 하나의 결측치를 Entity-Embedding Net 을 사용하여, 추정하였다.

이 작업을 나머지 결측치에도 동일하게 적용되어, 총 변수의 개수 만큼의 딥러닝 모델을 생성했다.

### 3. 앙상블

추정한 방법을 모두 앙상블을 통한 값을 최종 결과로서 

결측치를 추정하였다.
