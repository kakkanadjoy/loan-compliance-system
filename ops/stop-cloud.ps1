# Stop the billable cloud resources. The Container App self-stops (scale-to-zero).
az postgres flexible-server stop --resource-group rg-loan-compliance --name kakka-loan-db
Write-Host "Database stopped. Reminder: Azure auto-restarts stopped servers after 7 days."