#!/bin/bash

PROCS=(1 2 4 6 8 12)
EXEC="./main"

if [ ! -f "$EXEC" ]; then
    echo "Ошибка: Исполняемый файл $EXEC не найден!"
    exit 1
fi

echo "----------------------------------------"

for N in "${PROCS[@]}"; do
    echo "Запуск на N = $N процессах..."
    
    mpirun --use-hwthread-cpus -N "$N" "$EXEC" 0
    
    echo "----------------------------------------"
done

echo "=== Все тесты завершены ==="