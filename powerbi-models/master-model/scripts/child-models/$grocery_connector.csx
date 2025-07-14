var version = "$grocery_connector"; // Grocery connector specific model

// Set model-specific properties
Model.Database.ID = "Modern Data Stack Grocery Connector";
Model.Database.Name = "Modern Data Stack Grocery Connector";

// Set the connector model expression
Model.Expressions["ConnectorModel"].Expression = "\"grocery_connector\"";

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
    var tableId = t.GetAnnotation(version + ".BigQuery.TableId");

    if(tableId != null) t.SetAnnotation("bigquery_table_id", tableId);
    
    foreach(var annotationKey in t.Annotations.Keys.ToList()) {
        if (annotationKey.StartsWith("$") && annotationKey.EndsWith(".BigQuery.TableId"))
            t.RemoveAnnotation(annotationKey);
    }
}

// Apply grocery connector naming conventions for new tables
if (Model.Tables.ContainsName("Dim Store"))
{
    Model.Tables["Dim Store"].Name = "Dim Location";
}

if (Model.Tables.ContainsName("Fact Inventory"))
{
    Model.Tables["Fact Inventory"].Name = "Fact Inventory Location";
}

// Apply systematic Product → Item nomenclature transformation
foreach(Table t in Model.Tables) {
    // Rename tables
    if(t.Name == "Dim Product") {
        t.Name = "Dim Item";
    }
    else if(t.Name == "Dim Store") {
        t.Name = "Dim Location"; 
    }
    
    // Rename columns with Product → Item and Store → Location transformations
    foreach(Column c in t.Columns.ToList()) {
        var originalName = c.Name;
        var newName = originalName;
        
        // Product → Item transformations
        newName = newName.Replace("Product", "Item");
        newName = newName.Replace("product", "item");
        
        // Store → Location transformations  
        newName = newName.Replace("Store", "Location");
        newName = newName.Replace("store", "location");
        
        // Apply custom column name annotations if they exist
        var groceryColumnName = c.GetAnnotation(version + "_ColumnName");
        if(groceryColumnName != null) {
            newName = groceryColumnName;
        }
        
        if(newName != originalName) {
            c.Name = newName;
        }
        
        // Apply sourceColumn transformation for grocery connector
        var grocerySourceColumn = c.GetAnnotation(version + ".sourceColumn");
        if(grocerySourceColumn != null) {
            c.SourceColumn = grocerySourceColumn;
        }
    }
}

// Update all measure expressions to use transformed table and column names
foreach(Measure m in Model.AllMeasures) {
    var originalExpression = m.Expression;
    var newExpression = originalExpression;
    
    // Update table references
    newExpression = newExpression.Replace("'Dim Product'", "'Dim Item'");
    newExpression = newExpression.Replace("'Dim Store'", "'Dim Location'");
    newExpression = newExpression.Replace("'Fact Inventory'", "'Fact Inventory Location'");
    
    // Update column references for Product → Item
    newExpression = newExpression.Replace("[Product", "[Item");
    newExpression = newExpression.Replace("[product", "[item");
    
    // Update column references for Store → Location
    newExpression = newExpression.Replace("[Store", "[Location");
    newExpression = newExpression.Replace("[store", "[location");
    
    if(newExpression != originalExpression) {
        m.Expression = newExpression;
    }
}

// Update relationship names to reflect table name changes
foreach(Relationship r in Model.Relationships.ToList()) {
    var originalName = r.Name;
    var newName = originalName;
    
    // Update relationship names for Product → Item transformations
    newName = newName.Replace("Dim Product", "Dim Item");
    newName = newName.Replace("Dim Store", "Dim Location");
    newName = newName.Replace("Fact Inventory", "Fact Inventory Location");
    
    if(newName != originalName) {
        r.Name = newName;
    }
} 