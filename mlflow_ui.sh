#!/bin/bash
PORT=5000

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p|--port) PORT="$2"; shift ;;
        stop) STOP=1 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ "$STOP" == "1" ]; then
    echo "Stopping MLflow UI servers..."
    pkill -f "mlflow.server:app"
    echo "MLflow UI servers stopped."
    exit 0
fi

if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "Port $PORT is already in use."
    echo "Please choose a different port with -p option or stop existing servers with 'stop' command."
    echo "Usage: $0 [-p PORT] [stop]"
    exit 1
fi

echo "Starting MLflow UI server on port $PORT..."
echo "You can access the UI at http://localhost:$PORT"
echo "Press Ctrl+C to stop the server."
mlflow ui --port $PORT 