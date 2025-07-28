import pandas as pd

def summarize_supplier_delivery(
    delivery_csv_path,           
    supplier_master_csv_path,    
    output_csv_path              
):
    # Load data
    delivery = pd.read_csv(delivery_csv_path)
    master = pd.read_csv(supplier_master_csv_path)

    # Clean column names if needed
    delivery.columns = delivery.columns.str.strip()
    master.columns = master.columns.str.strip()

    # Group by Supplier ID and Value Category, aggregate numeric fields by mean
    summary = (
        delivery.groupby(['Supplier ID', 'Value Category'], as_index=False)
        .agg({
            '%_on_time': 'mean',
            '%_delayed': 'mean',
            'avg_delay': 'mean',
            'avg_shipment_volume': 'mean',
            '%_shipment_lost': 'mean' if '%_shipment_lost' in delivery.columns else 'mean',
            '%_defect_rate': 'mean' if '%_defect_rate' in delivery.columns else 'mean'
        })
    )

    # Merge with master to add Supplier Name
    summary = summary.merge(
        master[['Supplier ID', 'Supplier Name']],
        on='Supplier ID',
        how='left'
    )

    # Reorder columns (Supplier ID, Supplier Name, Value Category, ...)
    cols = ['Supplier ID', 'Supplier Name', 'Value Category'] + [col for col in summary.columns if col not in ['Supplier ID', 'Supplier Name', 'Value Category']]
    summary = summary[cols]

    # Sort by Value Category in the order: Critical, High, Medium, Low
    category_order = ['Critical', 'High', 'Medium', 'Low']
    summary['Value Category'] = pd.Categorical(summary['Value Category'], categories=category_order, ordered=True)
    summary = summary.sort_values('Value Category').reset_index(drop=True)


    # Save to CSV
    summary.to_csv(output_csv_path, index=False)
    print(f"supplier_delivery saved to {output_csv_path}")
