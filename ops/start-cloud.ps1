# Wake the cloud. Database takes ~3-5 min; the app wakes itself on first request.
az postgres flexible-server start --resource-group rg-loan-compliance --name kakka-loan-db
Write-Host "Database starting. App URL:"
az containerapp show --name loan-compliance-api --resource-group rg-loan-compliance --query properties.configuration.ingress.fqdn --output tsv