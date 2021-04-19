# Knowledge Distillation



* Ensemble 방법은 모델의 성능을 높여줄 수 있는 좋은 방법이지만 소모되는 컴퓨팅 파워와 메모리 공간 대비 얻어지는 성능의 효율이 굉장히 낮으며 이는 제한된 컴퓨팅 자원에서 최대한 효율적인 모델을 구축해야 할 때 걸림돌이 됨. 따라서 거대한 ensemble 모델(Teacher model)로 얻을 수 있는 일반화 능력을 상대적으로 작은 모델(Student model)에 증류하는 방법으로 성능 향상을 구현





* Methods

  * **Softer Softmax**

    기존의 Softmax는 exponential 연산으로 인해 1에 가까운 확률은 더 1로 가중되고 0에 가까운 값은 더 0으로 가중되는 특성이 존재. 이는 정답이 아닌 class의 정보(**dark knowledge**)를 축소시키는 단점이 있으므로 이를 완화한 Softer softmax 사용하여 정보가 잘 전달되도록 함

    

    

    ![Softmax](/image/그림1.png)

    T=1일 때 일반적인 Softmax 함수이며, T > 1일때 기존보다 더 완만한 distribution을 가지는 Softer softmax임  

  * **Soft label**

    기존의 one-hot Encoding 된 label을 hard label이라고 하며, Softer softmax의 결과로 나온 확률값을 Soft label이라고 함. 

    

    ![Softmax](/image/그림2.png)

    위 그림과 같이 Soft label은 데이터의 정보를 hard label보다 잘 반영함  

  * **Distillation loss**

    

    ![Softmax](/image/그림3.png)

    기존 학습은 모델의 softmax output과 one-hot label을 비교해 loss(1)를 계산했지만 distillation을 사용한 경우, 학습이 완료된 teacher model의 Softer softmax output과 student model의 Softer softmax output을 비교하여 얻은 loss(2)를 loss(1)과 적당한 비율로 더해 사용함
