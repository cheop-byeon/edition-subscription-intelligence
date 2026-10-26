CREATE OR REPLACE VIEW KKBOX_FEATURES AS

      WITH ranked AS (
        SELECT t.*, ROW_NUMBER() OVER (
          PARTITION BY t.dataset_id,t.msno
          ORDER BY t.transaction_date DESC,t.membership_expire_date DESC,
                   t.is_cancel ASC,t.plan_list_price DESC,t.actual_amount_paid DESC,
                   t.payment_plan_days DESC,t.is_auto_renew DESC) AS rn
        FROM kkbox_transactions t JOIN kkbox_labels l
          ON t.dataset_id=l.dataset_id AND t.msno=l.msno
        WHERE t.transaction_date < l.snapshot_date
      ), history AS (
        SELECT t.dataset_id,t.msno,MIN(t.transaction_date) AS first_date,
          SUM(CASE WHEN DATEDIFF('day',t.transaction_date,l.snapshot_date) <= 90 THEN 1 ELSE 0 END) AS transactions_90d,
          SUM(CASE WHEN DATEDIFF('day',t.transaction_date,l.snapshot_date) <= 90 THEN t.actual_amount_paid ELSE 0 END) AS paid_total_90d,
          AVG(CASE WHEN DATEDIFF('day',t.transaction_date,l.snapshot_date) <= 90 THEN t.actual_amount_paid ELSE NULL END) AS paid_mean_90d
        FROM kkbox_transactions t JOIN kkbox_labels l
          ON t.dataset_id=l.dataset_id AND t.msno=l.msno
        WHERE t.transaction_date < l.snapshot_date GROUP BY t.dataset_id,t.msno
      )
      SELECT l.dataset_id,l.msno,l.split_name,l.is_churn,
        h.transactions_90d,h.paid_total_90d,COALESCE(h.paid_mean_90d,0) AS paid_mean_90d,
        r.payment_plan_days AS last_plan_days,r.plan_list_price AS last_list_price,
        r.actual_amount_paid AS last_amount_paid,r.is_auto_renew AS last_auto_renew,
        r.is_cancel AS last_cancel_flag,
        DATEDIFF('day',r.transaction_date,l.snapshot_date) AS days_since_transaction,
        DATEDIFF('day',l.snapshot_date,r.membership_expire_date) AS days_to_expiry,
        DATEDIFF('day',h.first_date,l.snapshot_date) AS history_days,
        r.plan_list_price-r.actual_amount_paid AS last_discount
      FROM kkbox_labels l JOIN ranked r
        ON l.dataset_id=r.dataset_id AND l.msno=r.msno AND r.rn=1
      JOIN history h ON l.dataset_id=h.dataset_id AND l.msno=h.msno
    ;
