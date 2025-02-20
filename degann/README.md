Ключевая идея перевода на PyTorch - создать базовые абстрактные классы и интерфейсы, которые описывают общую логику для всех компонентов (лоссы, слои, оптимизаторы и т.д.). Эти абстракции можно назвать "базовыми классами", а каждая конкретная реализация (для TensorFlow или PyTorch) будет реализовывать или наследоваться от этих базовых классов.
Как правило, достаточной функциональстью будут обладать фабрики.

```
class Base(ABC):
    @abstractmethod
    def __call__(self, y_true, y_pred):
        pass

class TensorflowItem(BaseLoss):
    def __call__(self, y_true, y_pred):
        # Реализация для TensorFlow
        pass

class PyTorchItem(BaseLoss):
    def __call__(self, y_true, y_pred):
        # Реализация для PyTorch
        pass

class ItemFactory:
    def __init__(self, framework: str):
        self.framework = framework

    def get(self, name: str):
        # Реализация необходимого интерфейса
```
С помощью такого принципа реализации "корневых" модулей наследующие модули будут иметь общий интерфейс работы с ним. Например, если пользователю или разработчику понадобится взаимодействие с Loss функцией 

```
loss_factory = LossFactory(framework='tensorflow')
loss = loss_factory.get_loss('mse')
output = loss(y_true, y_pred)
``` 