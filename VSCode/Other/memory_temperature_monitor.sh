while true; do
    sensors | grep -A 2 'Mem'
    sleep 1
done

