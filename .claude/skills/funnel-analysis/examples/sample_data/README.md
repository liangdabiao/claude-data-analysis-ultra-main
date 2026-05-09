# Sample Data for Funnel Analysis

This directory contains sample datasets to help you understand how to structure your data for funnel analysis.

## Data Structure Requirements

For effective funnel analysis, your data should include:

### Required Columns
- **User Identifier**: Unique ID for each user (e.g., `user_id`, `customer_id`)
- **Step Indicators**: Boolean flags or timestamps for each funnel step

### Optional Columns (for segmentation)
- **Device Type**: `device`, `platform`, `client_type`
- **User Segment**: `segment`, `user_type`, `customer_tier`
- **Demographics**: `gender`, `age_group`, `location`
- **Temporal Data**: `date`, `timestamp`, `cohort`

## Example Data Formats

### E-commerce Funnel
```csv
user_id,device,segment,homepage,search,product_view,add_to_cart,purchase
1001,Mobile,New User,True,True,False,False,False
1002,Desktop,Returning User,True,True,True,True,True
1003,Mobile,VIP User,True,True,True,True,False
```

### Marketing Funnel
```csv
user_id,traffic_source,campaign,ad_click,landing_page,sign_up,email_verify
1001,Google,campaign_a,True,True,True,False
1002,Facebook,campaign_b,True,False,False,False
1003,Direct,organic,True,True,True,True
```

### User Onboarding Funnel
```csv
user_id,sign_up_date,profile_complete,tutorial_start,tutorial_complete,first_action
1001,2024-01-15,True,True,True,True
1002,2024-01-16,True,False,False,False
1003,2024-01-17,True,True,True,False
```

## Key Considerations

1. **Data Consistency**: Ensure consistent user identification across all steps
2. **Step Order**: Maintain logical order in your funnel steps
3. **Missing Values**: Handle missing data appropriately (usually means user didn't reach that step)
4. **Time Windows**: Consider appropriate time frames for user journeys
5. **Sample Size**: Ensure adequate sample sizes for reliable analysis

## Data Validation Checklist

- [ ] Unique user identifiers for all records
- [ ] Clear step definitions and indicators
- [ ] Consistent data types across columns
- [ ] Appropriate handling of missing values
- [ ] Sufficient sample size for each segment
- [ ] Logical flow between funnel steps