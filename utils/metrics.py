import numpy as np
import matplotlib.pyplot as plt
import os
import json

def plot_metrics(logs, save_path=None):
    """
    Plot training and testing metrics from logs
    
    Args:
        logs (dict): Dictionary containing training logs
        save_path (str, optional): Path to save plot
    """
    try:
        train_metrics = logs.get('train_metrics', [])
        test_metrics = logs.get('test_metrics', [])
        
        # Check if we have any metrics to plot
        if not train_metrics and not test_metrics:
            print("No metrics available to plot")
            return
        
        # Create a figure for accuracy and loss
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot training and testing accuracy
        if train_metrics:
            train_rounds = [m['round'] for m in train_metrics]
            train_accuracies = [m['accuracy'] for m in train_metrics]
            ax1.plot(train_rounds, train_accuracies, marker='o', label='Train')
        
        if test_metrics:
            test_rounds = [m['round'] for m in test_metrics]
            test_accuracies = [m['accuracy'] for m in test_metrics]
            ax1.plot(test_rounds, test_accuracies, marker='s', label='Test')
        
        ax1.set_xlabel('Rounds')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Training and Testing Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot training and testing loss
        if train_metrics:
            train_losses = [m['loss'] for m in train_metrics]
            ax2.plot(train_rounds, train_losses, marker='o', label='Train')
        
        if test_metrics:
            test_losses = [m['loss'] for m in test_metrics]
            ax2.plot(test_rounds, test_losses, marker='s', label='Test')
        
        ax2.set_xlabel('Rounds')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training and Testing Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Metrics plot saved to {save_path}")
            except Exception as e:
                print(f"Error saving plot to {save_path}: {e}")
        
        plt.show()
        
        # Plot client-specific metrics if available
        if 'client_metrics' in logs and logs['client_metrics']:
            domains = list(logs['client_metrics'].keys())
            
            if domains:
                try:
                    # Create a figure for client accuracies
                    plt.figure(figsize=(12, 6))
                    
                    # Plot accuracy for each client/domain
                    for domain in domains:
                        if not logs['client_metrics'][domain]:
                            continue
                            
                        client_rounds = [m['round'] for m in logs['client_metrics'][domain]]
                        client_accuracies = [m['accuracy'] for m in logs['client_metrics'][domain]]
                        
                        plt.plot(client_rounds, client_accuracies, marker='o', label=f'{domain}')
                    
                    plt.xlabel('Rounds')
                    plt.ylabel('Accuracy (%)')
                    plt.title('Client-Specific Accuracies by Domain')
                    plt.legend()
                    plt.grid(True)
                    
                    if save_path:
                        try:
                            client_path = save_path.replace('.png', '_clients.png')
                            plt.savefig(client_path, dpi=300, bbox_inches='tight')
                            print(f"Client metrics plot saved to {client_path}")
                        except Exception as e:
                            print(f"Error saving client plot: {e}")
                    
                    plt.show()
                except Exception as e:
                    print(f"Error plotting client-specific metrics: {e}")
    
    except Exception as e:
        print(f"Error plotting metrics: {e}")
        import traceback
        traceback.print_exc()

def plot_domain_performance(logs, save_path=None):
    """
    Plot performance comparison between domains
    
    Args:
        logs (dict): Dictionary containing training logs
        save_path (str, optional): Path to save the plot
    """
    if 'client_metrics' not in logs or not logs['client_metrics']:
        print("No client-specific metrics available to plot domain performance")
        return
        
    # Get final accuracy for each domain
    domains = []
    accuracies = []
    
    for domain, metrics in logs['client_metrics'].items():
        if metrics:
            domains.append(domain)
            # Get the last round's accuracy
            accuracies.append(metrics[-1]['accuracy'])
    
    if not domains:
        print("No domain data available")
        return
        
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(domains, accuracies, color='skyblue')
    
    # Add value labels on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, 
                 bar.get_height() + 1, 
                 f"{acc:.2f}%", 
                 ha='center')
    
    plt.xlabel('Domains')
    plt.ylabel('Final Accuracy (%)')
    plt.title('Final Accuracy by Domain')
    plt.ylim(0, max(accuracies) * 1.2)  # Add some space above bars for labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        domain_path = save_path.replace('.png', '_domains.png')
        plt.savefig(domain_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def load_logs(log_path):
    """
    Load training logs from a JSON file
    
    Args:
        log_path (str): Path to the JSON log file
        
    Returns:
        dict: Loaded logs
    """
    with open(log_path, 'r') as f:
        logs = json.load(f)
    return logs

def calculate_improvements(logs):
    """
    Calculate improvement metrics from logs
    
    Args:
        logs (dict): Dictionary containing training logs
        
    Returns:
        dict: Improvement metrics
    """
    try:
        train_metrics = logs.get('train_metrics', [])
        test_metrics = logs.get('test_metrics', [])
        
        # Initialize with default values
        improvements = {
            'train_accuracy_improvement': 0.0,
            'test_accuracy_improvement': 0.0,
            'train_loss_improvement': 0.0,
            'test_loss_improvement': 0.0,
            'domain_improvements': {}
        }
        
        # Calculate accuracy improvements if metrics are available
        if train_metrics and len(train_metrics) >= 2:
            initial_train_acc = train_metrics[0]['accuracy']
            final_train_acc = train_metrics[-1]['accuracy']
            improvements['train_accuracy_improvement'] = final_train_acc - initial_train_acc
            
            initial_train_loss = train_metrics[0]['loss']
            final_train_loss = train_metrics[-1]['loss']
            improvements['train_loss_improvement'] = initial_train_loss - final_train_loss
        
        if test_metrics and len(test_metrics) >= 2:
            initial_test_acc = test_metrics[0]['accuracy']
            final_test_acc = test_metrics[-1]['accuracy']
            improvements['test_accuracy_improvement'] = final_test_acc - initial_test_acc
            
            initial_test_loss = test_metrics[0]['loss']
            final_test_loss = test_metrics[-1]['loss']
            improvements['test_loss_improvement'] = initial_test_loss - final_test_loss
        
        # Calculate domain-specific improvements if available
        domain_improvements = {}
        
        if 'client_metrics' in logs and logs['client_metrics']:
            for domain, metrics in logs['client_metrics'].items():
                if metrics and len(metrics) >= 2:  # Need at least two points for improvement
                    try:
                        initial_acc = metrics[0]['accuracy']
                        final_acc = metrics[-1]['accuracy']
                        acc_improvement = final_acc - initial_acc
                        
                        domain_improvements[domain] = {
                            'initial_accuracy': initial_acc,
                            'final_accuracy': final_acc,
                            'improvement': acc_improvement
                        }
                    except (KeyError, IndexError) as e:
                        print(f"Error calculating improvements for domain {domain}: {e}")
        
        improvements['domain_improvements'] = domain_improvements
        return improvements
    
    except Exception as e:
        print(f"Error calculating improvements: {e}")
        import traceback
        traceback.print_exc()
        return {
            'train_accuracy_improvement': 0.0,
            'test_accuracy_improvement': 0.0,
            'train_loss_improvement': 0.0,
            'test_loss_improvement': 0.0,
            'domain_improvements': {}
        }

def print_summary(logs):
    """
    Print a summary of the training results
    
    Args:
        logs (dict): Dictionary containing training logs
    """
    try:
        train_metrics = logs.get('train_metrics', [])
        test_metrics = logs.get('test_metrics', [])
        
        print("\n=============== TRAINING SUMMARY ===============")
        
        if train_metrics and test_metrics:
            # Calculate improvements
            improvements = calculate_improvements(logs)
            
            print("\nAccuracy improvements:")
            print(f"  Train: {improvements['train_accuracy_improvement']:.2f}%")
            print(f"  Test: {improvements['test_accuracy_improvement']:.2f}%")
            
            if train_metrics:
                print("\nFinal train metrics:")
                print(f"  Accuracy: {train_metrics[-1]['accuracy']:.2f}%")
                print(f"  Loss: {train_metrics[-1]['loss']:.4f}")
            
            if test_metrics:
                print("\nFinal test metrics:")
                print(f"  Accuracy: {test_metrics[-1]['accuracy']:.2f}%")
                print(f"  Loss: {test_metrics[-1]['loss']:.4f}")
            
            # Print domain improvements if available
            if improvements.get('domain_improvements'):
                print("\nDomain-specific improvements:")
                for domain, domain_imp in improvements['domain_improvements'].items():
                    print(f"  {domain}: {domain_imp['improvement']:.2f}% "
                          f"({domain_imp['initial_accuracy']:.2f}% → {domain_imp['final_accuracy']:.2f}%)")
        else:
            print("\nNot enough data to calculate metrics.")
            
        print("\n================================================\n")
        
    except Exception as e:
        print(f"Error printing summary: {e}")
        import traceback
        traceback.print_exc()
        print("\nSummary unavailable due to errors in the logs.")
        print("\n================================================\n")
    
    # Plot domain comparison
    plot_domain_performance(logs) 