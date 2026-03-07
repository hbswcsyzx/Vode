"""
Level 5: Real-World Production Models
Reference to production-scale models for scalability and filtering tests.
"""

# This level references the unifolm-world-model-action project
# Located at: /home/zyf/XXX/unifolm-world-model-action

"""
Purpose:
--------
This level serves as the ultimate stress test for Vode's function-level tracing capabilities.
It tests:
- Scalability with large, production-scale models
- Filtering strategies for complex codebases
- torch.nn integration in real-world scenarios
- Performance under realistic workloads

Usage:
------
To trace the unifolm-world-model-action project:

    vode python /home/zyf/XXX/unifolm-world-model-action/[entry_script].py

Expected Challenges:
-------------------
1. Large number of function calls requiring aggressive filtering
2. Complex module hierarchies requiring proper torch.nn support
3. Third-party library calls requiring exclusion strategies
4. Multiple input/output formats requiring robust standardization
5. Large tensor values requiring efficient stats-only recording

Success Criteria:
----------------
- Vode should successfully trace the model without crashing
- Output graph should be navigable (not overwhelming)
- Key model components should be identifiable
- Function-level dataflow should be clear
- Performance overhead should be acceptable (< 10x slowdown)

Notes:
------
This is the most complex test case and may require:
- Tuning of max-depth parameter
- Enabling third-party filtering
- Adjusting value-policy settings
- Using file-path based filtering
"""


def main():
    """
    This file serves as documentation for Level 5 testing.
    
    Actual testing should be done by running Vode on the
    unifolm-world-model-action project directly.
    """
    print("Level 5: Real-World Production Models")
    print("=" * 60)
    print()
    print("This level references: /home/zyf/XXX/unifolm-world-model-action")
    print()
    print("To test, run Vode on the actual project:")
    print("  vode --max-depth 5 --exclude-third-party \\")
    print("       python /home/zyf/XXX/unifolm-world-model-action/[script].py")
    print()
    print("This will test:")
    print("  - Scalability with production models")
    print("  - Filtering effectiveness")
    print("  - torch.nn integration")
    print("  - Performance characteristics")
    print()


if __name__ == "__main__":
    main()
