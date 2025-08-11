
## Connect on the kube
```bash
kubectl -n lextral port-forward svc/lextral-postgresql 5433:5432
```

```bash
psql -h 127.0.0.1 -p 5433 -U app -d lextral-db
```
