#!/usr/bin/env python3
"""
Script to update DKSalaries CSV file with predictions from madden_predictions file.
Updates AvgPointsPerGame and adds new columns: fppg_floor, fppg_ceiling, min_exposure, max_exposure, min_deviation, max_deviation.
"""

import pandas as pd
import sys
import os
from pathlib import Path


def normalize_name(name):
    """Normalize player names for matching (lowercase, strip whitespace)."""
    if pd.isna(name):
        return ""
    return str(name).strip().lower()


def match_player(name, team, position, predictions_dict):
    """
    Match a player from salaries file to predictions file.
    Returns the prediction if found, None otherwise.
    """
    # Try exact match first
    key = (normalize_name(name), team.upper() if pd.notna(team) else "")
    if key in predictions_dict:
        return predictions_dict[key]
    
    # Try matching by name only (in case team doesn't match exactly)
    for (pred_name, pred_team), pred_data in predictions_dict.items():
        if pred_name == key[0]:
            return pred_data
    
    return None


def update_salaries_with_predictions(predictions_file, salaries_file):
    """
    Update salaries CSV with predictions.
    
    Args:
        predictions_file: Path to madden_predictions CSV file
        salaries_file: Path to DKSalaries CSV file (will be updated in place)
    """
    # Read predictions file
    print(f"Reading predictions from: {predictions_file}")
    pred_df = pd.read_csv(predictions_file)
    
    # Create lookup dictionary: (normalized_name, team) -> prediction_data
    predictions_dict = {}
    for _, row in pred_df.iterrows():
        player_name = normalize_name(row['Player'])
        team = str(row['TeamAbbrev']).upper().strip() if pd.notna(row['TeamAbbrev']) else ""
        key = (player_name, team)
        predictions_dict[key] = {
            'PredictedFPTS': row['PredictedFPTS'] if pd.notna(row['PredictedFPTS']) else 0.0,
            'Player': row['Player'],
            'TeamAbbrev': team
        }
    
    print(f"Loaded {len(predictions_dict)} predictions")
    
    # Read salaries file
    print(f"Reading salaries from: {salaries_file}")
    salaries_df = pd.read_csv(salaries_file)
    
    # Initialize new columns
    salaries_df['fppg_floor'] = 0.0
    salaries_df['fppg_ceiling'] = 0.0
    salaries_df['min_exposure'] = 0
    salaries_df['max_exposure'] = 100
    salaries_df['min_deviation'] = 0
    salaries_df['max_deviation'] = 0
    
    # Track matches for reporting
    matched_count = 0
    unmatched_players = []
    
    # Update each row
    for idx, row in salaries_df.iterrows():
        player_name = row['Name'] if pd.notna(row['Name']) else ""
        team = str(row['TeamAbbrev']).upper().strip() if pd.notna(row['TeamAbbrev']) else ""
        position = row['Position'] if pd.notna(row['Position']) else ""
        
        # Try to find matching prediction
        prediction = match_player(player_name, team, position, predictions_dict)
        
        if prediction:
            # Update AvgPointsPerGame with prediction
            salaries_df.at[idx, 'AvgPointsPerGame'] = prediction['PredictedFPTS']
            # Set ceiling to prediction * 1.5
            salaries_df.at[idx, 'fppg_ceiling'] = prediction['PredictedFPTS'] * 1.5
            matched_count += 1
        else:
            # No prediction found, set to 0
            salaries_df.at[idx, 'AvgPointsPerGame'] = 0.0
            salaries_df.at[idx, 'fppg_ceiling'] = 0.0
            unmatched_players.append(f"{player_name} ({team}, {position})")
    
    print(f"\nMatched {matched_count} out of {len(salaries_df)} players")
    if unmatched_players:
        print(f"\nUnmatched players ({len(unmatched_players)}):")
        for player in unmatched_players[:10]:  # Show first 10
            print(f"  - {player}")
        if len(unmatched_players) > 10:
            print(f"  ... and {len(unmatched_players) - 10} more")
    
    # Reorder columns as specified
    column_order = [
        'Position', 'Name + ID', 'Name', 'ID', 'Roster Position', 'Salary',
        'Game Info', 'TeamAbbrev', 'AvgPointsPerGame', 'fppg_floor', 'fppg_ceiling',
        'max_exposure', 'min_exposure', 'max_deviation', 'min_deviation'
    ]
    
    # Only include columns that exist in the dataframe
    final_columns = [col for col in column_order if col in salaries_df.columns]
    salaries_df = salaries_df[final_columns]
    salaries_df = salaries_df[salaries_df['AvgPointsPerGame'] > 2]
    # Write updated file
    print(f"\nWriting updated file to: {salaries_file}")
    salaries_df.to_csv(salaries_file, index=False)
    print("Done!")


def main():
    """Main function to handle command line arguments or use defaults."""
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    
    if len(sys.argv) >= 2:
        # Use provided predictions file path
        predictions_file = sys.argv[1]
    else:
        # Default: look for latest predictions file in data directory
        # You can customize this logic
        predictions_file = data_dir / 'madden_predictions_DKSalaries_20251121_late.csv'
    
    if len(sys.argv) >= 3:
        # Use provided salaries file path
        salaries_file = sys.argv[2]
    else:
        # Default: use salaries file in notebook directory
        salaries_file = script_dir / 'DKSalaries_20251121_late.csv'
    
    # Check if files exist
    if not os.path.exists(predictions_file):
        print(f"Error: Predictions file not found: {predictions_file}")
        sys.exit(1)
    
    if not os.path.exists(salaries_file):
        print(f"Error: Salaries file not found: {salaries_file}")
        sys.exit(1)
    
    update_salaries_with_predictions(predictions_file, salaries_file)


if __name__ == '__main__':
    main()



