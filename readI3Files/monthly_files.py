import subprocess
import calendar
import multiprocessing

def run_readI3_for_month(month):
    year = 2023
    start_day = 1
    start_month = month
    end_month = month
    end_day = calendar.monthrange(year, month)[1]
    command = [
        'python', 'readI3.py',
        '-stD', f'{start_day:02d}',
        '-stM', f'{start_month:02d}',
        '-enD', f'{end_day:02d}',
        '-enM', f'{end_month:02d}',
        '-y', str(year)
    ]
    print(f'Executing: {" ".join(command)}')
    subprocess.run(command)

def main():
    with multiprocessing.Pool(processes=12) as pool:
        pool.map(run_readI3_for_month, range(1, 13))

if __name__ == '__main__':
    main()
