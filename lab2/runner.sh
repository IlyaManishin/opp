#!/bin/bash

EXECUTABLES=("parallel2")
THREADS=(1 2 4 6 8 12 14)

for exe in "${EXECUTABLES[@]}"; do
    if [ ! -f "./$exe" ]; then
        echo "Предупреждение: Файл $exe не найден, пропускаю."
        continue
    fi

    output_file="${exe}-res.csv"
    echo "Threads,Exec_Time_Sec" > "$output_file"
    echo "Тестирование $exe..."

    for t in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$t

        echo -n "  Запуск с $t потоками... "
        duration=$(./"$exe")
        
        echo "${t},${duration}" >> "$output_file"
        echo "Готово ($duration сек)."
    done
    echo "Результаты сохранены в $output_file"
    echo "-----------------------------------"
done