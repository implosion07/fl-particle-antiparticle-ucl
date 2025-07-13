#!/bin/bash

echo "Starting FL server..."
gnome-terminal -- bash -c "python3 insur_FL_server.py; exec bash" &

sleep 20
python3 insur_maskedAgg.py &

echo "Starting FL clients..."
for i in {0..9}; do
    gnome-terminal -- bash -c "python3 insur_FL_client.py --agent_id=$i; exec bash" &
done

echo "All clients have been started."
