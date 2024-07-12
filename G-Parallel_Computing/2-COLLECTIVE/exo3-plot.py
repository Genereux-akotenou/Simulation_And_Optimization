import subprocess
import time
import matplotlib.pyplot as plt

initial_processes = 1
num_runs = 7

# Command to execute
command_template = "mpirun -n {num_processes} python3 exo3.py"

# Record execution times
execution_times = []
process = []

for i in range(num_runs):
    command = command_template.format(num_processes=initial_processes + i)

    # Execute the command and measure execution time
    start_time = time.time()
    subprocess.run(command, shell=True)
    end_time = time.time()

    # Calculate execution time and record
    execution_time = end_time - start_time
    execution_times.append(execution_time)
    process.append(initial_processes + i)

# Plot
print(process)
print(execution_times)
plt.plot(process, execution_times, label='Scalability')
plt.xlabel('Numer of process')
plt.ylabel('CPU Times')
plt.title('Scalability plot')
plt.show()
