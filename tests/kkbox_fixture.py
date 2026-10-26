"""Clearly synthetic official-schema fixture for integration tests, never benchmark data."""
import csv
import random
from pathlib import Path


def make_fixture(folder, count=2000):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(714)
    with (folder / 'train_v2.csv').open('w', newline='') as labels, (folder / 'transactions.csv').open('w', newline='') as transactions:
        label_writer = csv.writer(labels)
        transaction_writer = csv.writer(transactions)
        label_writer.writerow(['msno', 'is_churn'])
        transaction_writer.writerow(['msno', 'transaction_date', 'membership_expire_date', 'payment_plan_days', 'plan_list_price', 'actual_amount_paid', 'is_auto_renew', 'is_cancel'])
        for i in range(count):
            user = f'SYNTHETIC-TEST-ONLY-{i:06}'
            auto_renew = i % 2
            churn = int(randomizer.random() < (.08 if auto_renew else .65))
            label_writer.writerow([user, churn])
            for month in ['01', '02']:
                transaction_writer.writerow([user, f'2017{month}15', f'2017{month}28', 30, 149, 129 + i % 20, auto_renew, 0])
            # A future transaction deliberately encodes the target. A leaky
            # feature builder would pick it as the latest payment or renewal.
            transaction_writer.writerow([user, '20170315', '20170415', 30, 9999 * churn, 9999 * churn, 1 - churn, churn])
    return folder


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    make_fixture(args.folder)
    print('Synthetic integration-test files written; these are not real KKBOX records.')
