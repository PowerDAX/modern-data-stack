var version = "$retail_connector"; // Retail connector specific model

// Set model-specific properties
Model.Database.ID = "Modern Data Stack Retail Connector";
Model.Database.Name = "Modern Data Stack Retail Connector";

// Set the connector model expression
Model.Expressions["ConnectorModel"].Expression = "\"retail_connector\"";

// Remove tables, measures, columns and hierarchies that are not part of the perspective:
foreach(var t in Model.Tables.ToList()) {
    if(!t.InPerspective[version]) t.Delete();
    else {
        foreach(var m in t.Measures.ToList()) if(!m.InPerspective[version]) m.Delete();   
        foreach(var c in t.Columns.ToList()) if(!c.InPerspective[version]) c.Delete();
        foreach(var h in t.Hierarchies.ToList()) if(!h.InPerspective[version]) h.Delete();
    }
}

// Remove user perspectives based on annotations and all developer perspectives:
foreach(var p in Model.Perspectives.ToList()) {
    if(p.Name.StartsWith("$")) p.Delete();

    // Keep all other perspectives that do not have the "DevPerspectives" annotation, while removing
    // those that have the annotation, if <version> is not specified in the annotation:
    if(p.GetAnnotation("DevPerspectives") != null && !p.GetAnnotation("DevPerspectives").Contains(version)) 
        p.Delete();
}

// Remove data sources based on annotations:
foreach(var ds in Model.DataSources.ToList()) {
    if(ds.GetAnnotation("DevPerspectives") == null) continue;
    if(!ds.GetAnnotation("DevPerspectives").Contains(version)) ds.Delete();
}

// Remove roles based on annotations:
foreach(var r in Model.Roles.ToList()) {
    if(r.GetAnnotation("DevPerspectives") == null) continue;
    if(!r.GetAnnotation("DevPerspectives").Contains(version)) r.Delete();
}

// Modify measures based on annotations:
foreach(Measure m in Model.AllMeasures) {
    var expr = m.GetAnnotation(version + "_Expression"); if(expr == null) continue;
    m.Expression = expr;
    m.FormatString = m.GetAnnotation(version + "_FormatString");
    m.Description = m.GetAnnotation(version + "_Description");    
}

// Set partition queries according to annotations:
foreach(Table t in Model.Tables) {
    var queryWithPlaceholders = t.GetAnnotation(version + "_PartitionQuery"); if(queryWithPlaceholders == null) continue;
    
    // Loop through all partitions in this table:
    foreach(Partition p in t.Partitions) {
        
        var finalQuery = queryWithPlaceholders;

        // Replace all placeholder values:
        foreach(var placeholder in p.Annotations.Keys) {
            finalQuery = finalQuery.Replace("%" + placeholder + "%", p.GetAnnotation(placeholder));
        }

        p.Query = finalQuery;
    }
}

// Set the bigquery_table_id annotation according to annotations
foreach(Table t in Model.Tables) {
    var tableId = t.GetAnnotation(version + "_bigquery_table_id");

    if(tableId != null) t.SetAnnotation("bigquery_table_id", tableId);
    
    foreach(var annotationKey in t.Annotations.Keys.ToList()) {
        if (annotationKey.StartsWith("$") && annotationKey.EndsWith("_bigquery_table_id"))
            t.RemoveAnnotation(annotationKey);
    }
}

// Apply retail connector naming conventions for new tables  
if (Model.Tables.ContainsName("Fact Inventory"))
{
    Model.Tables["Fact Inventory"].Name = "Fact Inventory Store";
}

// Remove grocery-specific columns that don't exist in retail connector
foreach(Table t in Model.Tables) {
    if(t.Name == "Fact Sales") {
        // Remove grocery-specific promotional/regular sales columns
        var groceryOnlyColumns = new[] { 
            "regular_sales_dollars", 
            "regular_sales_units", 
            "promotional_sales_dollars", 
            "promotional_sales_units" 
        };
        
        foreach(var columnName in groceryOnlyColumns) {
            var column = t.Columns.FirstOrDefault(c => c.Name == columnName);
            if(column != null) column.Delete();
        }
    }
    
    if(t.Name == "Fact Inventory Store") {
        // Remove grocery-specific inventory columns
        var groceryOnlyColumns = new[] { 
            "available_quantity", 
            "allocated_quantity", 
            "allocated_units", 
            "available_units" 
        };
        
        foreach(var columnName in groceryOnlyColumns) {
            var column = t.Columns.FirstOrDefault(c => c.Name == columnName);
            if(column != null) column.Delete();
        }
    }
}

// Remove grocery-specific measures that don't apply to retail connector
foreach(Table t in Model.Tables) {
    if(t.Name == "Fact Sales") {
        var groceryOnlyMeasures = new[] { 
            "Regular Sales Amount", 
            "Regular Sales Quantity", 
            "Promotional Sales Amount", 
            "Promotional Sales Quantity" 
        };
        
        foreach(var measureName in groceryOnlyMeasures) {
            var measure = t.Measures.FirstOrDefault(m => m.Name == measureName);
            if(measure != null) measure.Delete();
        }
    }
}

// Clean up column annotations that are not needed for retail connector
foreach(Column c in Model.AllColumns) {
    // Remove grocery connector column name annotations since we use the default names
    c.RemoveAnnotation("$grocery_connector_ColumnName");
}

// Update all measure expressions to use transformed table names
foreach(Measure m in Model.AllMeasures) {
    var originalExpression = m.Expression;
    var newExpression = originalExpression;
    
    // Update table references for Fact Inventory → Fact Inventory Store
    newExpression = newExpression.Replace("'Fact Inventory'", "'Fact Inventory Store'");
    
    if(newExpression != originalExpression) {
        m.Expression = newExpression;
    }
}

// Update relationship names to reflect table name changes
foreach(Relationship r in Model.Relationships.ToList()) {
    var originalName = r.Name;
    var newName = originalName;
    
    // Update relationship names for Fact Inventory → Fact Inventory Store
    newName = newName.Replace("Fact Inventory", "Fact Inventory Store");
    
    if(newName != originalName) {
        r.Name = newName;
    }
} 