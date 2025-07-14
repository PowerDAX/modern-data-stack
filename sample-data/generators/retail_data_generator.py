#!/usr/bin/env python3
"""
Synthetic Retail Data Generator for Modern Data Stack Showcase
============================================================

This module generates realistic synthetic retail datasets with proper statistical
distributions, seasonal patterns, and business logic for demonstrating modern
data stack capabilities.

Features:
- Realistic sales transaction generation
- Inventory level simulation
- Product master data with hierarchies
- Store master data with geographic distribution
- Seasonal and trend patterns
- Data quality issues simulation
- Performance metrics generation

Author: Data Engineering Team
Date: 2024-01-15
"""

import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import random
from dataclasses import dataclass, field
from faker import Faker
import json
import os
from pathlib import Path

# Configure random seeds for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()
fake.seed_instance(42)

@dataclass
class GenerationConfig:
    """Configuration for data generation"""
    # Time range
    start_date: datetime = datetime(2022, 1, 1)
    end_date: datetime = datetime(2024, 1, 1)
    
    # Volume configuration
    num_stores: int = 50
    num_products: int = 500
    num_customers: int = 10000
    transactions_per_day: int = 1000
    
    # Business logic parameters
    seasonal_factor: float = 0.3
    trend_factor: float = 0.1
    promotion_frequency: float = 0.15
    stockout_probability: float = 0.05
    
    # Data quality parameters
    missing_data_rate: float = 0.02
    duplicate_rate: float = 0.01
    outlier_rate: float = 0.005
    
    # Output configuration
    output_format: str = 'csv'
    chunk_size: int = 10000
    include_data_quality_issues: bool = True

class RetailDataGenerator:
    """
    Comprehensive retail data generator with realistic patterns and distributions.
    
    This class generates synthetic retail data including:
    - Product master data with hierarchies
    - Store master data with geographic distribution
    - Sales transactions with seasonal patterns
    - Inventory levels with realistic turnover
    - Performance metrics and KPIs
    """
    
    def __init__(self, config: GenerationConfig = None):
        """
        Initialize the retail data generator.
        
        Parameters
        ----------
        config : GenerationConfig, optional
            Configuration for data generation
        """
        self.config = config or GenerationConfig()
        self.fake = Faker()
        self.fake.seed_instance(42)
        
        # Initialize data structures
        self.products_df = None
        self.stores_df = None
        self.customers_df = None
        self.categories_df = None
        self.suppliers_df = None
        
        # Pre-generate reference data
        self._initialize_reference_data()
    
    def _initialize_reference_data(self):
        """Initialize reference data for consistent generation"""
        # Product categories
        self.categories = [
            'Electronics', 'Clothing', 'Home & Garden', 'Sports & Outdoors',
            'Health & Beauty', 'Books & Media', 'Toys & Games', 'Food & Beverages',
            'Automotive', 'Office Supplies', 'Pet Supplies', 'Jewelry & Watches'
        ]
        
        # Product subcategories
        self.subcategories = {
            'Electronics': ['Smartphones', 'Laptops', 'Tablets', 'Accessories', 'Gaming'],
            'Clothing': ['Mens', 'Womens', 'Kids', 'Shoes', 'Accessories'],
            'Home & Garden': ['Furniture', 'Appliances', 'Tools', 'Decor', 'Gardening'],
            'Sports & Outdoors': ['Fitness', 'Outdoor Gear', 'Team Sports', 'Water Sports'],
            'Health & Beauty': ['Skincare', 'Cosmetics', 'Supplements', 'Personal Care'],
            'Books & Media': ['Fiction', 'Non-Fiction', 'Movies', 'Music', 'Games'],
            'Toys & Games': ['Action Figures', 'Board Games', 'Educational', 'Electronic'],
            'Food & Beverages': ['Packaged Foods', 'Beverages', 'Snacks', 'Organic']
        }
        
        # Store types
        self.store_types = ['Flagship', 'Standard', 'Outlet', 'Express', 'Mall']
        
        # Regions
        self.regions = ['North', 'South', 'East', 'West', 'Central']
        
        # Suppliers
        self.suppliers = [
            'TechCorp Solutions', 'Fashion Forward Inc', 'Home Essentials Ltd',
            'Sports World Co', 'Beauty First Corp', 'Media Masters Inc',
            'Toy Universe Ltd', 'Fresh Foods Co', 'Auto Parts Direct',
            'Office Pro Supply', 'Pet Paradise Inc', 'Luxury Goods Ltd'
        ]
    
    def generate_product_master(self) -> pd.DataFrame:
        """
        Generate product master data with realistic hierarchies and attributes.
        
        Returns
        -------
        pd.DataFrame
            Product master data with hierarchies
        """
        print("Generating product master data...")
        
        products = []
        product_id = 1
        
        for category in self.categories:
            # Generate 30-50 products per category
            num_products_in_category = random.randint(30, 50)
            
            for _ in range(num_products_in_category):
                subcategory = random.choice(self.subcategories[category])
                brand = self.fake.company()
                
                # Generate product attributes
                product = {
                    'product_id': f"P{product_id:06d}",
                    'product_name': self._generate_product_name(category, subcategory),
                    'category': category,
                    'subcategory': subcategory,
                    'brand': brand,
                    'supplier': random.choice(self.suppliers),
                    'unit_cost': round(random.uniform(5, 500), 2),
                    'unit_price': 0,  # Will be calculated based on markup
                    'weight_kg': round(random.uniform(0.1, 10.0), 2),
                    'dimensions_cm': f"{random.randint(10, 100)}x{random.randint(10, 100)}x{random.randint(5, 50)}",
                    'color': random.choice(['Red', 'Blue', 'Green', 'Black', 'White', 'Gray', 'Brown', 'Multi']),
                    'size': random.choice(['XS', 'S', 'M', 'L', 'XL', 'XXL', 'One Size', 'N/A']),
                    'material': random.choice(['Cotton', 'Plastic', 'Metal', 'Wood', 'Glass', 'Fabric', 'Leather', 'Synthetic']),
                    'is_active': random.choice([True, True, True, True, False]),  # 80% active
                    'launch_date': self.fake.date_between(start_date='-2y', end_date='today'),
                    'discontinue_date': None,
                    'rating': round(random.uniform(1.0, 5.0), 1),
                    'review_count': random.randint(0, 1000),
                    'is_organic': category == 'Food & Beverages' and random.choice([True, False]),
                    'is_premium': random.choice([True, False]),
                    'season': random.choice(['Spring', 'Summer', 'Fall', 'Winter', 'All Season']),
                    'created_date': datetime.now(),
                    'updated_date': datetime.now()
                }
                
                # Calculate unit price with markup
                markup_factor = random.uniform(1.5, 3.0)
                if product['is_premium']:
                    markup_factor *= 1.5
                product['unit_price'] = round(product['unit_cost'] * markup_factor, 2)
                
                # Set discontinue date for inactive products
                if not product['is_active']:
                    product['discontinue_date'] = self.fake.date_between(
                        start_date=product['launch_date'], 
                        end_date='today'
                    )
                
                products.append(product)
                product_id += 1
        
        self.products_df = pd.DataFrame(products)
        print(f"Generated {len(self.products_df)} products across {len(self.categories)} categories")
        return self.products_df
    
    def generate_store_master(self) -> pd.DataFrame:
        """
        Generate store master data with realistic geographic distribution.
        
        Returns
        -------
        pd.DataFrame
            Store master data with geographic information
        """
        print("Generating store master data...")
        
        stores = []
        
        for store_id in range(1, self.config.num_stores + 1):
            region = random.choice(self.regions)
            store_type = random.choice(self.store_types)
            
            # Generate store size based on type
            size_multiplier = {
                'Flagship': 2.0,
                'Standard': 1.0,
                'Outlet': 0.8,
                'Express': 0.5,
                'Mall': 1.2
            }[store_type]
            
            store_size = int(random.uniform(1000, 5000) * size_multiplier)
            
            store = {
                'store_id': f"S{store_id:04d}",
                'store_name': f"{region} {store_type} Store {store_id}",
                'store_type': store_type,
                'region': region,
                'district': f"{region} District {random.randint(1, 5)}",
                'city': self.fake.city(),
                'state': self.fake.state(),
                'country': 'USA',
                'address': self.fake.address(),
                'postal_code': self.fake.zipcode(),
                'latitude': round(random.uniform(25.0, 49.0), 6),
                'longitude': round(random.uniform(-125.0, -67.0), 6),
                'phone': self.fake.phone_number(),
                'email': f"store{store_id}@retaildemo.com",
                'manager_name': self.fake.name(),
                'open_date': self.fake.date_between(start_date='-5y', end_date='-6m'),
                'store_size_sqft': store_size,
                'employee_count': random.randint(5, 50),
                'parking_spaces': random.randint(20, 200),
                'has_pharmacy': random.choice([True, False]),
                'has_gas_station': random.choice([True, False]),
                'has_auto_service': random.choice([True, False]),
                'operating_hours': f"{random.randint(6, 9)}:00 AM - {random.randint(9, 11)}:00 PM",
                'is_active': random.choice([True, True, True, True, False]),  # 80% active
                'rent_per_sqft': round(random.uniform(15, 50), 2),
                'utilities_cost_monthly': round(random.uniform(1000, 5000), 2),
                'created_date': datetime.now(),
                'updated_date': datetime.now()
            }
            
            stores.append(store)
        
        self.stores_df = pd.DataFrame(stores)
        print(f"Generated {len(self.stores_df)} stores across {len(self.regions)} regions")
        return self.stores_df
    
    def generate_customer_master(self) -> pd.DataFrame:
        """
        Generate customer master data with realistic demographics.
        
        Returns
        -------
        pd.DataFrame
            Customer master data
        """
        print("Generating customer master data...")
        
        customers = []
        
        for customer_id in range(1, self.config.num_customers + 1):
            # Generate customer demographics
            gender = random.choice(['M', 'F', 'Other'])
            age = random.randint(18, 80)
            income_bracket = random.choice(['Low', 'Medium', 'High', 'Premium'])
            
            # Generate customer segments
            segment = random.choice(['New', 'Regular', 'VIP', 'Inactive'])
            
            customer = {
                'customer_id': f"C{customer_id:08d}",
                'first_name': self.fake.first_name(),
                'last_name': self.fake.last_name(),
                'email': self.fake.email(),
                'phone': self.fake.phone_number(),
                'gender': gender,
                'age': age,
                'birth_date': self.fake.date_of_birth(minimum_age=18, maximum_age=80),
                'address': self.fake.address(),
                'city': self.fake.city(),
                'state': self.fake.state(),
                'postal_code': self.fake.zipcode(),
                'country': 'USA',
                'income_bracket': income_bracket,
                'customer_segment': segment,
                'loyalty_member': random.choice([True, False]),
                'loyalty_points': random.randint(0, 10000) if random.choice([True, False]) else 0,
                'preferred_contact': random.choice(['Email', 'Phone', 'SMS', 'Mail']),
                'marketing_consent': random.choice([True, False]),
                'registration_date': self.fake.date_between(start_date='-3y', end_date='today'),
                'last_purchase_date': self.fake.date_between(start_date='-1y', end_date='today'),
                'total_lifetime_value': round(random.uniform(100, 10000), 2),
                'average_order_value': round(random.uniform(25, 500), 2),
                'purchase_frequency': random.randint(1, 50),
                'is_active': random.choice([True, True, True, False]),  # 75% active
                'created_date': datetime.now(),
                'updated_date': datetime.now()
            }
            
            customers.append(customer)
        
        self.customers_df = pd.DataFrame(customers)
        print(f"Generated {len(self.customers_df)} customers")
        return self.customers_df
    
    def generate_sales_transactions(self, num_days: int = None) -> pd.DataFrame:
        """
        Generate sales transactions with realistic patterns and seasonality.
        
        Parameters
        ----------
        num_days : int, optional
            Number of days to generate transactions for
            
        Returns
        -------
        pd.DataFrame
            Sales transaction data
        """
        print("Generating sales transactions...")
        
        if num_days is None:
            num_days = (self.config.end_date - self.config.start_date).days
        
        transactions = []
        transaction_id = 1
        
        # Generate transactions for each day
        for day_offset in range(num_days):
            current_date = self.config.start_date + timedelta(days=day_offset)
            
            # Calculate seasonal and trend factors
            seasonal_factor = self._calculate_seasonal_factor(current_date)
            trend_factor = self._calculate_trend_factor(current_date)
            
            # Adjust daily transaction volume
            base_transactions = self.config.transactions_per_day
            adjusted_transactions = int(base_transactions * seasonal_factor * trend_factor)
            
            # Generate transactions for the day
            for _ in range(adjusted_transactions):
                # Select random store and customer
                store_id = random.choice(self.stores_df['store_id'].tolist())
                customer_id = random.choice(self.customers_df['customer_id'].tolist())
                
                # Generate transaction items
                num_items = random.choices([1, 2, 3, 4, 5], weights=[40, 30, 15, 10, 5])[0]
                
                transaction_total = 0
                transaction_items = []
                
                for item_num in range(num_items):
                    product = self.products_df.sample(1).iloc[0]
                    
                    # Calculate quantity and price
                    quantity = random.randint(1, 5)
                    unit_price = product['unit_price']
                    
                    # Apply promotions randomly
                    if random.random() < self.config.promotion_frequency:
                        discount_rate = random.uniform(0.1, 0.5)
                        unit_price *= (1 - discount_rate)
                    
                    line_total = quantity * unit_price
                    transaction_total += line_total
                    
                    # Create transaction item
                    item = {
                        'transaction_id': f"T{transaction_id:010d}",
                        'item_number': item_num + 1,
                        'product_id': product['product_id'],
                        'quantity': quantity,
                        'unit_price': round(unit_price, 2),
                        'line_total': round(line_total, 2),
                        'discount_amount': round(product['unit_price'] * quantity - line_total, 2),
                        'tax_amount': round(line_total * 0.0875, 2)  # 8.75% tax
                    }
                    
                    transaction_items.append(item)
                
                # Create transaction header
                transaction = {
                    'transaction_id': f"T{transaction_id:010d}",
                    'store_id': store_id,
                    'customer_id': customer_id,
                    'transaction_date': current_date,
                    'transaction_time': self._generate_transaction_time(current_date),
                    'cashier_id': f"E{random.randint(1, 100):04d}",
                    'payment_method': random.choice(['Cash', 'Credit Card', 'Debit Card', 'Mobile Pay', 'Gift Card']),
                    'subtotal': round(transaction_total, 2),
                    'tax_amount': round(transaction_total * 0.0875, 2),
                    'discount_amount': sum(item['discount_amount'] for item in transaction_items),
                    'total_amount': round(transaction_total * 1.0875, 2),
                    'item_count': num_items,
                    'loyalty_points_earned': int(transaction_total * 0.1) if random.choice([True, False]) else 0,
                    'loyalty_points_redeemed': random.randint(0, 500) if random.random() < 0.1 else 0,
                    'is_return': random.choice([True, False]) if random.random() < 0.05 else False,
                    'return_reason': random.choice(['Defective', 'Wrong Size', 'Changed Mind', 'Gift Return']) if random.random() < 0.05 else None,
                    'created_date': datetime.now(),
                    'updated_date': datetime.now()
                }
                
                transactions.append(transaction)
                transaction_id += 1
        
        sales_df = pd.DataFrame(transactions)
        print(f"Generated {len(sales_df)} sales transactions over {num_days} days")
        return sales_df
    
    def generate_inventory_data(self) -> pd.DataFrame:
        """
        Generate inventory data with realistic stock levels and turnover.
        
        Returns
        -------
        pd.DataFrame
            Inventory data
        """
        print("Generating inventory data...")
        
        inventory_records = []
        
        # Generate inventory for each store-product combination
        for store_id in self.stores_df['store_id']:
            # Each store carries 60-80% of all products
            products_in_store = self.products_df.sample(
                n=int(len(self.products_df) * random.uniform(0.6, 0.8))
            )
            
            for _, product in products_in_store.iterrows():
                # Calculate inventory levels based on product characteristics
                base_stock = self._calculate_base_stock(product)
                
                # Generate inventory snapshots for different dates
                for days_back in [0, 7, 14, 30, 60, 90]:
                    snapshot_date = datetime.now() - timedelta(days=days_back)
                    
                    # Calculate current stock level
                    current_stock = max(0, base_stock + random.randint(-20, 20))
                    
                    # Simulate stockouts
                    if random.random() < self.config.stockout_probability:
                        current_stock = 0
                    
                    inventory_record = {
                        'store_id': store_id,
                        'product_id': product['product_id'],
                        'snapshot_date': snapshot_date,
                        'current_stock': current_stock,
                        'reserved_stock': random.randint(0, min(5, current_stock)),
                        'available_stock': current_stock - random.randint(0, min(5, current_stock)),
                        'reorder_point': random.randint(5, 20),
                        'reorder_quantity': random.randint(25, 100),
                        'max_stock_level': base_stock + random.randint(10, 30),
                        'unit_cost': product['unit_cost'],
                        'total_value': round(current_stock * product['unit_cost'], 2),
                        'days_supply': random.randint(1, 30) if current_stock > 0 else 0,
                        'turnover_rate': round(random.uniform(4, 12), 2),
                        'last_received_date': self.fake.date_between(start_date='-30d', end_date='today'),
                        'last_sold_date': self.fake.date_between(start_date='-7d', end_date='today'),
                        'supplier_id': random.choice(self.suppliers),
                        'warehouse_location': f"Aisle {random.randint(1, 20)}, Shelf {random.randint(1, 10)}",
                        'created_date': datetime.now(),
                        'updated_date': datetime.now()
                    }
                    
                    inventory_records.append(inventory_record)
        
        inventory_df = pd.DataFrame(inventory_records)
        print(f"Generated {len(inventory_df)} inventory records")
        return inventory_df
    
    def generate_performance_metrics(self) -> pd.DataFrame:
        """
        Generate performance metrics and KPIs.
        
        Returns
        -------
        pd.DataFrame
            Performance metrics data
        """
        print("Generating performance metrics...")
        
        metrics = []
        
        # Generate daily performance metrics for each store
        for store_id in self.stores_df['store_id']:
            for days_back in range(365):
                metric_date = datetime.now() - timedelta(days=days_back)
                
                # Calculate base metrics
                daily_sales = random.uniform(1000, 10000)
                daily_customers = random.randint(50, 500)
                daily_transactions = random.randint(40, 400)
                
                metric = {
                    'store_id': store_id,
                    'metric_date': metric_date,
                    'daily_sales': round(daily_sales, 2),
                    'daily_customers': daily_customers,
                    'daily_transactions': daily_transactions,
                    'average_transaction_value': round(daily_sales / daily_transactions, 2),
                    'conversion_rate': round(daily_transactions / daily_customers, 3),
                    'items_per_transaction': round(random.uniform(1.5, 4.0), 2),
                    'gross_margin': round(daily_sales * random.uniform(0.25, 0.45), 2),
                    'labor_cost': round(random.uniform(500, 2000), 2),
                    'operating_expenses': round(random.uniform(200, 800), 2),
                    'net_profit': 0,  # Will be calculated
                    'foot_traffic': random.randint(100, 1000),
                    'peak_hour_sales': round(daily_sales * random.uniform(0.15, 0.25), 2),
                    'customer_satisfaction': round(random.uniform(3.5, 5.0), 2),
                    'employee_productivity': round(random.uniform(80, 120), 2),
                    'inventory_turnover': round(random.uniform(6, 15), 2),
                    'shrinkage_rate': round(random.uniform(0.5, 3.0), 3),
                    'return_rate': round(random.uniform(2, 8), 2),
                    'created_date': datetime.now(),
                    'updated_date': datetime.now()
                }
                
                # Calculate net profit
                metric['net_profit'] = round(
                    metric['gross_margin'] - metric['labor_cost'] - metric['operating_expenses'], 2
                )
                
                metrics.append(metric)
        
        metrics_df = pd.DataFrame(metrics)
        print(f"Generated {len(metrics_df)} performance metric records")
        return metrics_df
    
    def _generate_product_name(self, category: str, subcategory: str) -> str:
        """Generate realistic product names based on category"""
        category_prefixes = {
            'Electronics': ['Pro', 'Ultra', 'Smart', 'Digital', 'Wireless'],
            'Clothing': ['Premium', 'Classic', 'Modern', 'Casual', 'Formal'],
            'Home & Garden': ['Deluxe', 'Essential', 'Comfort', 'Luxury', 'Practical'],
            'Sports & Outdoors': ['Athletic', 'Performance', 'Outdoor', 'Sport', 'Active'],
            'Health & Beauty': ['Natural', 'Organic', 'Premium', 'Essential', 'Pure'],
            'Food & Beverages': ['Fresh', 'Organic', 'Artisan', 'Gourmet', 'Natural']
        }
        
        prefixes = category_prefixes.get(category, ['Premium', 'Quality', 'Standard'])
        prefix = random.choice(prefixes)
        
        # Generate a base name
        base_name = f"{prefix} {subcategory} {random.choice(['Pro', 'Max', 'Plus', 'Elite', 'Standard'])}"
        
        return base_name
    
    def _calculate_seasonal_factor(self, date: datetime) -> float:
        """Calculate seasonal factor for sales volume"""
        month = date.month
        
        # Higher sales in November-December (holidays)
        if month in [11, 12]:
            return 1.0 + self.config.seasonal_factor
        # Lower sales in January-February
        elif month in [1, 2]:
            return 1.0 - self.config.seasonal_factor * 0.5
        # Spring boost in March-May
        elif month in [3, 4, 5]:
            return 1.0 + self.config.seasonal_factor * 0.3
        # Summer variability
        elif month in [6, 7, 8]:
            return 1.0 + self.config.seasonal_factor * 0.2
        # Fall preparation
        else:
            return 1.0 + self.config.seasonal_factor * 0.1
    
    def _calculate_trend_factor(self, date: datetime) -> float:
        """Calculate trend factor for sales growth"""
        days_from_start = (date - self.config.start_date).days
        total_days = (self.config.end_date - self.config.start_date).days
        
        # Linear growth trend
        trend = 1.0 + (self.config.trend_factor * days_from_start / total_days)
        
        # Add some randomness
        trend *= random.uniform(0.9, 1.1)
        
        return trend
    
    def _generate_transaction_time(self, date: datetime) -> datetime:
        """Generate realistic transaction times"""
        # Business hours: 8 AM to 10 PM
        hour = random.choices(
            range(8, 23),
            weights=[1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3]  # Peak around lunch and evening
        )[0]
        
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        return datetime.combine(date.date(), datetime.min.time().replace(hour=hour, minute=minute, second=second))
    
    def _calculate_base_stock(self, product: pd.Series) -> int:
        """Calculate base stock level for a product"""
        # Base stock depends on product category and price
        if product['category'] == 'Electronics':
            return random.randint(10, 50)
        elif product['category'] == 'Clothing':
            return random.randint(20, 100)
        elif product['category'] == 'Food & Beverages':
            return random.randint(50, 200)
        else:
            return random.randint(15, 75)
    
    def introduce_data_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Introduce realistic data quality issues for testing and validation.
        
        Parameters
        ----------
        df : pd.DataFrame
            Clean dataset
            
        Returns
        -------
        pd.DataFrame
            Dataset with introduced quality issues
        """
        if not self.config.include_data_quality_issues:
            return df
        
        df_copy = df.copy()
        
        # Introduce missing values
        for column in df_copy.columns:
            if df_copy[column].dtype in ['object', 'string']:
                # Random missing values
                mask = np.random.random(len(df_copy)) < self.config.missing_data_rate
                df_copy.loc[mask, column] = None
        
        # Introduce duplicates
        if len(df_copy) > 10:
            duplicate_indices = np.random.choice(
                df_copy.index, 
                size=int(len(df_copy) * self.config.duplicate_rate),
                replace=False
            )
            df_copy = pd.concat([df_copy, df_copy.loc[duplicate_indices]])
        
        # Introduce outliers in numeric columns
        numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            outlier_mask = np.random.random(len(df_copy)) < self.config.outlier_rate
            outlier_values = df_copy[column].std() * np.random.normal(0, 5, outlier_mask.sum())
            df_copy.loc[outlier_mask, column] = outlier_values
        
        return df_copy
    
    def generate_full_dataset(self, output_dir: str = "generated_data") -> Dict[str, pd.DataFrame]:
        """
        Generate the complete retail dataset.
        
        Parameters
        ----------
        output_dir : str
            Directory to save generated data
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing all generated datasets
        """
        print("🚀 Starting full retail dataset generation...")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate all datasets
        datasets = {}
        
        # Master data
        datasets['products'] = self.generate_product_master()
        datasets['stores'] = self.generate_store_master()
        datasets['customers'] = self.generate_customer_master()
        
        # Transaction data
        datasets['sales'] = self.generate_sales_transactions()
        datasets['inventory'] = self.generate_inventory_data()
        datasets['performance'] = self.generate_performance_metrics()
        
        # Introduce data quality issues
        if self.config.include_data_quality_issues:
            print("Introducing data quality issues...")
            for name, df in datasets.items():
                datasets[name] = self.introduce_data_quality_issues(df)
        
        # Save datasets
        for name, df in datasets.items():
            if self.config.output_format == 'csv':
                output_path = Path(output_dir) / f"{name}.csv"
                df.to_csv(output_path, index=False)
                print(f"Saved {name}: {len(df)} records -> {output_path}")
            elif self.config.output_format == 'parquet':
                output_path = Path(output_dir) / f"{name}.parquet"
                df.to_parquet(output_path, index=False)
                print(f"Saved {name}: {len(df)} records -> {output_path}")
        
        # Generate data summary
        self._generate_data_summary(datasets, output_dir)
        
        print(f"✅ Dataset generation complete! Files saved to: {output_dir}")
        return datasets
    
    def _generate_data_summary(self, datasets: Dict[str, pd.DataFrame], output_dir: str):
        """Generate a summary of the generated datasets"""
        summary = {
            'generation_config': {
                'start_date': self.config.start_date.isoformat(),
                'end_date': self.config.end_date.isoformat(),
                'num_stores': self.config.num_stores,
                'num_products': self.config.num_products,
                'num_customers': self.config.num_customers,
                'transactions_per_day': self.config.transactions_per_day,
                'include_data_quality_issues': self.config.include_data_quality_issues
            },
            'datasets': {}
        }
        
        for name, df in datasets.items():
            summary['datasets'][name] = {
                'record_count': len(df),
                'column_count': len(df.columns),
                'columns': df.columns.tolist(),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                'data_types': df.dtypes.to_dict()
            }
        
        # Save summary
        summary_path = Path(output_dir) / "data_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Data summary saved to: {summary_path}")

def main():
    """Main function to generate retail datasets"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic retail datasets")
    parser.add_argument("--output-dir", default="generated_data", help="Output directory")
    parser.add_argument("--num-stores", type=int, default=50, help="Number of stores")
    parser.add_argument("--num-products", type=int, default=500, help="Number of products")
    parser.add_argument("--num-customers", type=int, default=10000, help="Number of customers")
    parser.add_argument("--transactions-per-day", type=int, default=1000, help="Transactions per day")
    parser.add_argument("--format", choices=['csv', 'parquet'], default='csv', help="Output format")
    parser.add_argument("--include-quality-issues", action="store_true", help="Include data quality issues")
    
    args = parser.parse_args()
    
    # Create configuration
    config = GenerationConfig(
        num_stores=args.num_stores,
        num_products=args.num_products,
        num_customers=args.num_customers,
        transactions_per_day=args.transactions_per_day,
        output_format=args.format,
        include_data_quality_issues=args.include_quality_issues
    )
    
    # Generate datasets
    generator = RetailDataGenerator(config)
    datasets = generator.generate_full_dataset(args.output_dir)
    
    print(f"\n🎉 Successfully generated {len(datasets)} datasets!")
    print(f"Total records: {sum(len(df) for df in datasets.values()):,}")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main() 