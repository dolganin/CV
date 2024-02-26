from imports import *

def balanced_train_test_split(dataset, test_prob):
    class_to_idx = dataset.class_to_idx
    
    # Создаем словарь для хранения индексов изображений для каждого класса
    class_indices = defaultdict(list)
    for idx, (_, class_idx) in enumerate(dataset.imgs):
        class_indices[class_idx].append(idx)
    
    # Разделяем индексы для train и test так, чтобы сохранить баланс классов
    train_indices = []
    test_indices = []
    for class_idx, indices in class_indices.items():
        # Разделяем индексы для каждого класса
        class_train_indices, class_test_indices = train_test_split(indices, test_size=test_prob, random_state=42)
        
        # Добавляем индексы в общий список для train и test
        train_indices.extend(class_train_indices)
        test_indices.extend(class_test_indices)
    
    # Создаем подмножества на основе индексов
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    return train_subset, test_subset