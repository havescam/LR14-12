# ==== Установка и импорты (ячейка Colab) ====
!pip install tensorflow --quiet

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class AdversarialAttack:
    def __init__(self, model):
        self.model = model

    def fgsm_attack(self, image, label, epsilon=0.01):
        image = tf.convert_to_tensor(image)
        label = tf.convert_to_tensor(label)
        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = self.model(image)
            loss = tf.keras.losses.sparse_categorical_crossentropy(label, prediction)
        gradient = tape.gradient(loss, image)
        signed_grad = tf.sign(gradient)
        adversarial_image = image + epsilon * signed_grad
        adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
        return adversarial_image.numpy()

    def pgd_attack(self, image, label, epsilon=0.01, alpha=0.01, num_iter=10):
        image = tf.convert_to_tensor(image)
        label = tf.convert_to_tensor(label)
        original = image
        for _ in range(num_iter):
            with tf.GradientTape() as tape:
                tape.watch(image)
                prediction = self.model(image)
                loss = tf.keras.losses.sparse_categorical_crossentropy(label, prediction)
            gradient = tape.gradient(loss, image)
            signed_grad = tf.sign(gradient)
            image = image + alpha * signed_grad
            image = tf.clip_by_value(image - original, -epsilon, epsilon) + original
            image = tf.clip_by_value(image, 0, 1)
        return image.numpy()

    def evaluate_robustness(self, test_images, test_labels, epsilon=0.01):
        original_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        adv_acc_fgsm = tf.keras.metrics.SparseCategoricalAccuracy()
        adv_acc_pgd = tf.keras.metrics.SparseCategoricalAccuracy()
        
        for image, label in zip(test_images, test_labels):
            pred_original = self.model.predict(np.expand_dims(image, 0))
            original_acc.update_state(label, pred_original)
            
            adv_fgsm = self.fgsm_attack(image, label, epsilon)
            pred_fgsm = self.model.predict(np.expand_dims(adv_fgsm, 0))
            adv_acc_fgsm.update_state(label, pred_fgsm)
            
            adv_pgd = self.pgd_attack(image, label, epsilon)
            pred_pgd = self.model.predict(np.expand_dims(adv_pgd, 0))
            adv_acc_pgd.update_state(label, pred_pgd)
        
        return {
            'original_accuracy': original_acc.result().numpy(),
            'fgsm_accuracy': adv_acc_fgsm.result().numpy(),
            'pgd_accuracy': adv_acc_pgd.result().numpy()
        }

    def plot_loss_and_accuracy(self):
        """
        Графики устойчивости модели к атакам
        1) Точность на оригинальных данных vs FGSM
        2) Точность на оригинальных данных vs PGD
        """
        print("Графики устойчивости модели к adversarial атакам")
        print("Требуется предварительное тестирование модели")

# ==== Загрузка данных ====
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# ==== Подготовка модели ====
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ==== Обучение модели ====
model.fit(x_train, y_train,
          epochs=5,
          batch_size=128,
          validation_split=0.1,
          verbose=1)

# ==== Создание атакующего объекта ====
attacker = AdversarialAttack(model)

# ==== Предсказания устойчивости ====
test_subset = x_test[:100]
test_subset_labels = y_test[:100]

results = attacker.evaluate_robustness(test_subset, test_subset_labels, epsilon=0.1)

# ==== Визуализация предсказаний ====
plt.figure(figsize=(14, 6))
metrics = ['original_accuracy', 'fgsm_accuracy', 'pgd_accuracy']
values = [results['original_accuracy'], results['fgsm_accuracy'], results['pgd_accuracy']]
plt.bar(metrics, values)
plt.title('Устойчивость модели к adversarial атакам')
plt.ylabel('Точность')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.show()

# ==== Графики точности ====
attacker.plot_loss_and_accuracy()
