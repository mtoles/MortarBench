import copy
import json
import os
import argparse
from pathlib import Path
from typing import Any
import logging
import random
from pydantic import BaseModel
from pydantic import Field


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Asset(BaseModel):
    sequence_number: int = Field(1, description="The sequence number of the asset")
    account_identifier: str = Field(
        "1212", description="The account identifier of the asset"
    )
    cash_or_market_value_amount: str = Field(
        "1000.00", description="The cash or market value of the asset"
    )
    asset_type: str = Field("CheckingAccount", description="The type of the asset")
    holder_full_name: str = Field(
        "Capital One", description="The full name of the holder of the asset"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ASSET_DETAIL": {
                "AssetAccountIdentifier": self.account_identifier,
                "AssetCashOrMarketValueAmount": self.cash_or_market_value_amount,
                "AssetType": self.asset_type,
            },
            "ASSET_HOLDER": {
                "NAME": {
                    "FullName": self.holder_full_name,
                }
            },
            "_SequenceNumber": str(self.sequence_number),
            "_xlink:label": f"ASSET_{self.sequence_number}",
        }


class Collateral(BaseModel):
    # does not relate to the bank statement
    address_line_text: str = Field(
        "1011 Brookside Ave", description="Property street address"
    )
    city_name: str = Field("Asbury Park", description="Property city")
    country_code: str = Field("US", description="Property country code")
    postal_code: str = Field("07712", description="Property postal code")
    state_code: str = Field("NJ", description="Property state code")
    fips_county_code: str = Field("025", description="FIPS county code")
    fips_state_numeric_code: str = Field("34", description="FIPS state numeric code")
    attachment_type: str = Field("Detached", description="Property attachment type")
    financed_unit_count: str = Field("1", description="Number of financed units")
    property_estimated_value_amount: str = Field(
        "1000000.00", description="Estimated property value"
    )
    property_valuation_amount: str = Field(
        "1000000.00", description="Appraised property valuation"
    )
    sales_contract_amount: str = Field(
        "1000000.00", description="Sales contract amount"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "SUBJECT_PROPERTY": {
                "ADDRESS": {
                    "AddressLineText": self.address_line_text,
                    "CityName": self.city_name,
                    "CountryCode": self.country_code,
                    "PostalCode": self.postal_code,
                    "StateCode": self.state_code,
                },
                "LOCATION_IDENTIFIER": {
                    "FIPS_INFORMATION": {
                        "FIPSCountyCode": self.fips_county_code,
                        "FIPSStateNumericCode": self.fips_state_numeric_code,
                    }
                },
                "PROPERTY_DETAIL": {
                    "AttachmentType": self.attachment_type,
                    "CommunityPropertyStateIndicator": "false",
                    "ConstructionMethodType": "SiteBuilt",
                    "FHASecondaryResidenceIndicator": "false",
                    "FinancedUnitCount": self.financed_unit_count,
                    "PropertyEstateType": "FeeSimple",
                    "PropertyEstimatedValueAmount": self.property_estimated_value_amount,
                    "PropertyExistingCleanEnergyLienIndicator": "false",
                    "PropertyInProjectIndicator": "false",
                    "PropertyMixedUsageIndicator": "false",
                    "PUDIndicator": "false",
                },
                "PROPERTY_VALUATIONS": {
                    "PROPERTY_VALUATION": {
                        "PROPERTY_VALUATION_DETAIL": {
                            "PropertyValuationAmount": self.property_valuation_amount,
                        }
                    }
                },
                "SALES_CONTRACTS": {
                    "SALES_CONTRACT": {
                        "SALES_CONTRACT_DETAIL": {
                            "SalesContractAmount": self.sales_contract_amount,
                        }
                    }
                },
            }
        }


class Liability(BaseModel):
    # Maybe related to the bank statement
    sequence_number: int = Field(1, description="The sequence number of the liability")
    account_identifier: str = Field(
        "3563A019732", description="The account identifier of the liability"
    )
    monthly_payment_amount: str = Field("123.00", description="Monthly payment amount")
    liability_type: str = Field("Installment", description="Liability type")
    unpaid_balance_amount: str = Field("2600.00", description="Unpaid balance amount")
    holder_full_name: str = Field(
        "MOUNTAIN BANK", description="Liability holder full name"
    )
    remaining_term_months_count: str = Field(
        "5", description="Remaining term in months"
    )

    def _liability_types(self) -> list[str]:
        liability_types = [
            "BNPL transactions",
            "child support",
            "undisclosed debt",
            "unexplained deposits",
            "secured loan",
        ]
        return liability_types

    def to_dict(self) -> dict[str, Any]:
        return {
            "LIABILITY_DETAIL": {
                "LiabilityAccountIdentifier": self.account_identifier,
                "LiabilityExclusionIndicator": "false",
                "LiabilityMonthlyPaymentAmount": self.monthly_payment_amount,
                "LiabilityPayoffStatusIndicator": "false",
                "LiabilityRemainingTermMonthsCount": self.remaining_term_months_count,
                "LiabilityType": self.liability_type,
                "LiabilityUnpaidBalanceAmount": self.unpaid_balance_amount,
            },
            "LIABILITY_HOLDER": {
                "NAME": {
                    "FullName": self.holder_full_name,
                }
            },
            "_SequenceNumber": str(self.sequence_number),
            "_xlink:label": f"LIABILITY_{self.sequence_number}",
        }


class Loan(BaseModel):
    # Maybe related to the bank statement
    sequence_number: int = Field(1, description="The sequence number of the loan")
    loan_identifier: str = Field("84192307554", description="Universal loan identifier")
    purchase_credit_amount: str = Field("5000.00", description="Purchase credit amount")
    base_loan_amount: str = Field("800000.00", description="Base loan amount")
    lien_priority_type: str = Field("FirstLien", description="Lien priority type")
    loan_purpose_type: str = Field("Purchase", description="Loan purpose type")
    mortgage_type: str = Field("Conventional", description="Mortgage type")
    note_rate_percent: str = Field("0.000", description="Loan note rate percent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "AFFORDABLE_LENDING": "",
            "CLOSING_INFORMATION": {
                "CLOSING_ADJUSTMENT_ITEMS": "",
                "CLOSING_INFORMATION_DETAIL": "",
            },
            "DOCUMENT_SPECIFIC_DATA_SETS": {
                "DOCUMENT_SPECIFIC_DATA_SET": {
                    "URLA": {
                        "URLA_DETAIL": "",
                    }
                }
            },
            "HOUSING_EXPENSES": "",
            "LOAN_DETAIL": {
                "BalloonIndicator": "false",
                "BelowMarketSubordinateFinancingIndicator": "false",
                "BorrowerCount": "1",
                "BuydownTemporarySubsidyFundingIndicator": "false",
                "ConstructionLoanIndicator": "false",
                "EnergyRelatedImprovementsIndicator": "false",
                "InterestOnlyIndicator": "false",
                "NegativeAmortizationIndicator": "false",
                "PrepaymentPenaltyIndicator": "false",
                "RenovationLoanIndicator": "false",
                "TotalMortgagedPropertiesCount": "1",
            },
            "LOAN_IDENTIFIERS": {
                "LOAN_IDENTIFIER": {
                    "LoanIdentifier": self.loan_identifier,
                    "LoanIdentifierType": "UniversalLoan",
                }
            },
            "ORIGINATION_SYSTEMS": {
                "ORIGINATION_SYSTEM": {
                    "LoanOriginationSystemName": "TidalWave SOLO",
                    "LoanOriginationSystemVendorIdentifier": "TidalWave",
                    "LoanOriginationSystemVersionIdentifier": "1.0",
                }
            },
            "PURCHASE_CREDITS": {
                "PURCHASE_CREDIT": {
                    "PurchaseCreditAmount": self.purchase_credit_amount,
                    "PurchaseCreditType": "EarnestMoney",
                }
            },
            "TERMS_OF_LOAN": {
                "BaseLoanAmount": self.base_loan_amount,
                "LienPriorityType": self.lien_priority_type,
                "LoanPurposeType": self.loan_purpose_type,
                "MortgageType": self.mortgage_type,
                "NoteRatePercent": self.note_rate_percent,
            },
            "_SequenceNumber": str(self.sequence_number),
            "_LoanRoleType": "SubjectLoan",
            "_xlink:label": f"LOAN_{self.sequence_number}",
        }


class Party(BaseModel):
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError()


class BorrowerParty(Party):
    sequence_number: int = Field(1, description="Borrower party sequence number")
    first_name: str = Field("Alice", description="Borrower first name")
    last_name: str = Field("Firstimer", description="Borrower last name")
    mobile_phone: str = Field("9999999999", description="Borrower mobile phone number")
    email: str = Field("borrower@example.com", description="Borrower email address")
    borrower_birth_date: str = Field("2000-01-01", description="Borrower birth date")
    dependent_count: str = Field("0", description="Borrower dependent count")
    marital_status_type: str = Field("Unmarried", description="Borrower marital status")
    current_income_monthly_total_amount: str = Field(
        "6678", description="Monthly income total"
    )
    employer_full_name: str = Field("NY MTA", description="Borrower employer full name")
    employment_start_date: str = Field(
        "2000-01-01", description="Employment start date"
    )
    employment_time_in_line_of_work_months_count: str = Field(
        "120", description="Months in line of work"
    )
    residence_address_line_text: str = Field(
        "9991 Warford St", description="Current residence address"
    )
    residence_city_name: str = Field("Dawson", description="Current residence city")
    residence_country_code: str = Field(
        "US", description="Current residence country code"
    )
    residence_postal_code: str = Field(
        "50066", description="Current residence postal code"
    )
    residence_state_code: str = Field("IA", description="Current residence state code")
    monthly_rent_amount: str = Field(
        "3000.00", description="Current monthly rent amount"
    )
    borrower_residency_duration_months_count: str = Field(
        "120", description="Residency duration in months"
    )
    taxpayer_identifier_value: str = Field(
        "991919991", description="Borrower SSN value"
    )
    borrower_residency_basis_type: str = Field(
        "Rent", description="Residency basis type"
    )
    borrower_residency_type: str = Field("Current", description="Residency type")
    citizenship_residency_type: str = Field(
        "USCitizen", description="Citizenship or residency type"
    )
    intent_to_occupy_type: str = Field(
        "Yes", description="Intent to occupy subject property"
    )
    homeowner_past_three_years_type: str = Field(
        "No", description="Homeowner in past three years"
    )
    employment_classification_type: str = Field(
        "Secondary", description="Employment classification"
    )
    employment_position_description: str = Field(
        "Worker", description="Employment position description"
    )
    employment_status_type: str = Field("Current", description="Employment status")
    party_role_type: str = Field("Borrower", description="Party role type")

    def to_dict(self) -> dict[str, Any]:
        return {
            "INDIVIDUAL": {
                "CONTACT_POINTS": {
                    "CONTACT_POINT": [
                        {
                            "CONTACT_POINT_TELEPHONE": {
                                "ContactPointTelephoneValue": self.mobile_phone
                            },
                            "CONTACT_POINT_DETAIL": {"ContactPointRoleType": "Mobile"},
                        },
                        {"CONTACT_POINT_EMAIL": {"ContactPointEmailValue": self.email}},
                    ]
                },
                "NAME": {
                    "FirstName": self.first_name,
                    "LastName": self.last_name,
                },
            },
            "ROLES": {
                "ROLE": {
                    "BORROWER": {
                        "BORROWER_DETAIL": {
                            "BorrowerBirthDate": self.borrower_birth_date,
                            "CommunityPropertyStateResidentIndicator": "false",
                            "DependentCount": self.dependent_count,
                            "MaritalStatusType": self.marital_status_type,
                        },
                        "CURRENT_INCOME": {
                            "CURRENT_INCOME_ITEMS": {
                                "CURRENT_INCOME_ITEM": {
                                    "CURRENT_INCOME_ITEM_DETAIL": {
                                        "CurrentIncomeMonthlyTotalAmount": self.current_income_monthly_total_amount,
                                        "EmploymentIncomeIndicator": "true",
                                        "IncomeType": "Base",
                                    },
                                    "_SequenceNumber": "1",
                                    "_xlink:label": f"CURRENT_INCOME_ITEM_{self.sequence_number}_1",
                                }
                            }
                        },
                        "DECLARATION": {
                            "DECLARATION_DETAIL": {
                                "BankruptcyIndicator": "false",
                                "CitizenshipResidencyType": self.citizenship_residency_type,
                                "HomeownerPastThreeYearsType": self.homeowner_past_three_years_type,
                                "IntentToOccupyType": self.intent_to_occupy_type,
                                "OutstandingJudgmentsIndicator": "false",
                                "PartyToLawsuitIndicator": "false",
                                "PresentlyDelinquentIndicator": "false",
                                "PriorPropertyDeedInLieuConveyedIndicator": "false",
                                "PriorPropertyForeclosureCompletedIndicator": "false",
                                "PriorPropertyShortSaleCompletedIndicator": "false",
                                "PropertyProposedCleanEnergyLienIndicator": "false",
                                "UndisclosedBorrowedFundsIndicator": "false",
                                "UndisclosedComakerOfNoteIndicator": "false",
                                "UndisclosedCreditApplicationIndicator": "false",
                                "UndisclosedMortgageApplicationIndicator": "false",
                                "EXTENSION": {
                                    "OTHER": {
                                        "DECLARATION_DETAIL_EXTENSION": {
                                            "SpecialBorrowerSellerRelationshipIndicator": {
                                                "__prefix": "ULAD",
                                                "__text": "false",
                                            },
                                            "__prefix": "ULAD",
                                        }
                                    }
                                },
                            }
                        },
                        "DEPENDENTS": "",
                        "EMPLOYERS": {
                            "EMPLOYER": {
                                "LEGAL_ENTITY": {
                                    "LEGAL_ENTITY_DETAIL": {
                                        "FullName": self.employer_full_name
                                    }
                                },
                                "EMPLOYMENT": {
                                    "EmploymentBorrowerSelfEmployedIndicator": "false",
                                    "EmploymentClassificationType": self.employment_classification_type,
                                    "EmploymentPositionDescription": self.employment_position_description,
                                    "EmploymentStartDate": self.employment_start_date,
                                    "EmploymentStatusType": self.employment_status_type,
                                    "EmploymentTimeInLineOfWorkMonthsCount": self.employment_time_in_line_of_work_months_count,
                                    "SpecialBorrowerEmployerRelationshipIndicator": "false",
                                    "EXTENSION": {
                                        "OTHER": {
                                            "EMPLOYMENT_EXTENSION": {
                                                "ForeignIncomeIndicator": {
                                                    "__prefix": "DU",
                                                    "__text": "false",
                                                },
                                                "SeasonalIncomeIndicator": {
                                                    "__prefix": "DU",
                                                    "__text": "false",
                                                },
                                                "__prefix": "DU",
                                            }
                                        }
                                    },
                                },
                                "_SequenceNumber": "1",
                                "_xlink:label": f"EMPLOYER_{self.sequence_number}_1",
                            }
                        },
                        "GOVERNMENT_BORROWER": "",
                        "GOVERNMENT_MONITORING": {
                            "GOVERNMENT_MONITORING_DETAIL": {
                                "HMDAEthnicityCollectedBasedOnVisualObservationOrSurnameIndicator": "false",
                                "HMDAEthnicityRefusalIndicator": "false",
                                "HMDAGenderCollectedBasedOnVisualObservationOrNameIndicator": "false",
                                "HMDAGenderRefusalIndicator": "false",
                                "HMDARaceCollectedBasedOnVisualObservationOrSurnameIndicator": "false",
                                "HMDARaceRefusalIndicator": "false",
                                "EXTENSION": {
                                    "OTHER": {
                                        "GOVERNMENT_MONITORING_DETAIL_EXTENSION": {
                                            "__prefix": "ULAD"
                                        }
                                    }
                                },
                            }
                        },
                        "RESIDENCES": {
                            "RESIDENCE": {
                                "ADDRESS": {
                                    "AddressLineText": self.residence_address_line_text,
                                    "CityName": self.residence_city_name,
                                    "CountryCode": self.residence_country_code,
                                    "PostalCode": self.residence_postal_code,
                                    "StateCode": self.residence_state_code,
                                },
                                "LANDLORD": {
                                    "LANDLORD_DETAIL": {
                                        "MonthlyRentAmount": self.monthly_rent_amount
                                    }
                                },
                                "RESIDENCE_DETAIL": {
                                    "BorrowerResidencyBasisType": self.borrower_residency_basis_type,
                                    "BorrowerResidencyDurationMonthsCount": self.borrower_residency_duration_months_count,
                                    "BorrowerResidencyType": self.borrower_residency_type,
                                },
                            }
                        },
                    },
                    "ROLE_DETAIL": {"PartyRoleType": self.party_role_type},
                    "_SequenceNumber": self.sequence_number,
                    "_xlink:label": f"BORROWER_{self.sequence_number}",
                }
            },
            "TAXPAYER_IDENTIFIERS": {
                "TAXPAYER_IDENTIFIER": {
                    "TaxpayerIdentifierType": "SocialSecurityNumber",
                    "TaxpayerIdentifierValue": self.taxpayer_identifier_value,
                }
            },
        }


class PropertyOwnerParty(Party):
    sequence_number: int = Field(1, description="Property owner party sequence number")
    property_owner_status_type: str = Field(
        "Proposed", description="Property owner status type"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "INDIVIDUAL": {"NAME": ""},
            "ROLES": {
                "ROLE": {
                    "PROPERTY_OWNER": {
                        "PropertyOwnerStatusType": self.property_owner_status_type
                    },
                    "ROLE_DETAIL": {"PartyRoleType": "PropertyOwner"},
                    "_SequenceNumber": str(self.sequence_number),
                    "_xlink:label": f"PROPERTY_OWNER_{self.sequence_number}",
                }
            },
        }


class Relationship(BaseModel):
    sequence_number: int = Field(1, description="Relationship sequence number")
    xlink_arcrole: str = Field(
        "urn:fdc:mismo.org:2009:residential/ASSET_IsAssociatedWith_ROLE",
        description="xlink arcrole",
    )
    xlink_from: str = Field("ASSET_1", description="xlink from label")
    xlink_to: str = Field("BORROWER_1", description="xlink to label")

    def to_dict(self) -> dict[str, Any]:
        return {
            "_SequenceNumber": str(self.sequence_number),
            "_xlink:arcrole": self.xlink_arcrole,
            "_xlink:from": self.xlink_from,
            "_xlink:to": self.xlink_to,
        }


class UladGenerator:
    """
    Generate ULAD JSON from a fixed template and only override values.

    Override keys can use dot + index notation, e.g.:
    - "MESSAGE.ABOUT_VERSIONS.ABOUT_VERSION.CreatedDatetime"
    - "MESSAGE.DEAL_SETS.DEAL_SET.DEALS.DEAL.ASSETS.ASSET[0].ASSET_DETAIL.AssetCashOrMarketValueAmount"
    """

    def __init__(self, template_path: str | Path, plaid_data: dict[str, Any], output_dir: str | Path | None = None):
        self.template_path = Path(template_path)
        self.template = self._load_template(self.template_path)
        self.plaid_data = plaid_data
        self.output_dir = output_dir

    def _load_template(self, template_path: Path) -> dict[str, Any]:
        with template_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def generate_assets(self) -> list[Asset]:
        # connect with the bank statement
        assets = []

        if not self.plaid_data:
            # print a warning
            logger.warning("No Plaid data provided, using static assets.")
            return [
                Asset(
                    sequence_number=1,
                    account_identifier="1212",
                    cash_or_market_value_amount="1000.00",
                    asset_type="CheckingAccount",
                    holder_full_name="Capital One",
                )
            ]
        
        else:
            for idx, account in enumerate(self.plaid_data["override_accounts"]):
                if account["type"] == "depository":
                    assets.append(
                        Asset(
                            sequence_number=idx + 1,
                            account_identifier=str(account["numbers"]["account"]),
                            cash_or_market_value_amount=str(account.get("end_balance", 0)),
                            asset_type="CheckingAccount" if account["subtype"] == "checking" else "SavingsAccount",
                            holder_full_name="Capital One",
                        )
                    )
                else:
                    continue
            return assets
                

    def generate_collateral(self) -> Collateral:
        # static for now
        # TODO: randomly generate a collateral
        return Collateral(
            sequence_number=1,
            address_line_text="1011 Brookside Ave",
            city_name="Asbury Park",
            country_code="US",
            postal_code="07712",
            state_code="NJ",
            fips_county_code="025",
            fips_state_numeric_code="34",
            attachment_type="Detached",
            financed_unit_count="1",
            property_estimated_value_amount="1000000.00",
            property_valuation_amount="1000000.00",
            sales_contract_amount="1000000.00",
        )

    def generate_liabilities(self) -> list[Liability]:
        # static for now
        # TODO: randomly generate a liability
        if not self.plaid_data:
            # print a warning
            logger.warning("No Plaid data provided, using static liabilities.")
            return [
                Liability(
                    sequence_number=1,
                    account_identifier="3563A019732",
                    monthly_payment_amount="123.00",
                    liability_type="Installment",
                    unpaid_balance_amount="2600.00",
                    holder_full_name="Mountain Bank",
                    remaining_term_months_count="5",
                )
            ]
        else:
            liabilities = []
            liability_types = Liability._liability_types()
            for idx, account in enumerate(self.plaid_data["override_accounts"]):
                transactions = account["transactions"]
                for transaction in transactions:
                    tag = transaction["tag"]
                    if tag in liability_types:
                        liability_type = tag
                        monthly_payment_amount = abs(transaction["amount"])
                        unpaid_months_count = random.randint(1, 12)
                    else:
                        continue
                    liabilities.append(
                        Liability(
                            sequence_number=idx + 1,
                            account_identifier=str(account["numbers"]["account"]),
                            monthly_payment_amount=str(monthly_payment_amount),
                            liability_type=liability_type,
                            unpaid_balance_amount=str(monthly_payment_amount * unpaid_months_count),
                            holder_full_name="Mountain Bank",
                            remaining_term_months_count=unpaid_months_count,
                        )
                    )
            return liabilities


    def generate_loans(self) -> list[Loan]:
        # static for now
        # TODO: randomly generate a loan
        if not self.plaid_data:
            # print a warning
            logger.warning("No Plaid data provided, using static assets.")
            return [
                Loan(
                    sequence_number=1,
                    loan_identifier="84192307554",
                    purchase_credit_amount="5000.00",
                    base_loan_amount="800000.00",
                    lien_priority_type="FirstLien",
                    loan_purpose_type="Purchase",
                    mortgage_type="Conventional",
                    note_rate_percent="0.000",
                )
            ]
        else:
            loans = []
            for idx, account in enumerate(self.plaid_data["override_accounts"]):
                transactions = account["transactions"]
                for transaction in transactions:
                    if "Earnest Money Deposit" in transaction["description"]:
                        purchase_credit_amount = abs(transaction["amount"])
                        # randomly generate a base loan amount above purchase_credit_amount
                        base_loan_amount = random.uniform(purchase_credit_amount * 1.1, purchase_credit_amount * 1.5)
                    loans.append(
                        Loan(
                            sequence_number=idx + 1,
                            loan_identifier=str(account["numbers"]["account"]),
                            purchase_credit_amount=str(purchase_credit_amount),
                            base_loan_amount=str(base_loan_amount),
                        )
                    )
            return loans

    def generate_parties(self) -> tuple[BorrowerParty, PropertyOwnerParty]:
        borrower_party = self.generate_borrower_party()
        property_owner_party = self.generate_property_owner_party()
        return borrower_party, property_owner_party

    def generate_relationships(
        self,
        assets: list[Asset],
        liabilities: list[Liability],
        borrower_party: BorrowerParty,
    ) -> list[Relationship]:
        relationships = []
        for asset in assets:
            relationships.append(
                Relationship(
                    sequence_number=len(relationships) + 1,
                    xlink_arcrole="urn:fdc:mismo.org:2009:residential/ASSET_IsAssociatedWith_ROLE",
                    xlink_from=f"ASSET_{asset.sequence_number}",
                    xlink_to=f"BORROWER_{borrower_party.sequence_number}",
                )
            )
        for liability in liabilities:
            relationships.append(
                Relationship(
                    sequence_number=len(relationships) + 1,
                    xlink_arcrole="urn:fdc:mismo.org:2009:residential/LIABILITY_IsAssociatedWith_ROLE",
                    xlink_from=f"LIABILITY_{liability.sequence_number}",
                    xlink_to=f"BORROWER_{borrower_party.sequence_number}",
                )
            )
        return relationships

    def generate_borrower_party(self) -> BorrowerParty:
        # static for now
        # TODO: randomly generate rest of values based on the plaid data
        if not self.plaid_data:
            logger.warning("No Plaid data provided, using static borrower party.")
            return BorrowerParty(
                sequence_number=1,
                first_name="Alice",
                last_name="Firstimer",
                mobile_phone="9999999999",
                email="borrower@example.com",
                borrower_birth_date="2000-01-01",
                dependent_count="0",
                marital_status_type="Unmarried",
                current_income_monthly_total_amount="6678",
                employer_full_name="NY MTA",
                employment_start_date="2000-01-01",
                employment_time_in_line_of_work_months_count="120",
                residence_address_line_text="9991 Warford St",
                residence_city_name="Dawson",
                residence_country_code="US",
                residence_postal_code="50066",
                residence_state_code="IA",
                monthly_rent_amount="3000.00",
                borrower_residency_duration_months_count="120",
                taxpayer_identifier_value="991919991",
                borrower_residency_basis_type="Rent",
                borrower_residency_type="Current",
                citizenship_residency_type="USCitizen",
                intent_to_occupy_type="Yes",
                homeowner_past_three_years_type="No",
                employment_classification_type="Secondary",
                employment_position_description="Worker",
                employment_status_type="Current",
                party_role_type="Borrower",
            )
        else:
            identity = self.plaid_data["override_accounts"][0]["identity"]
            name = identity["names"][0]
            email = identity["emails"][0]["data"]
            address = identity["addresses"][0]["data"]
            country = address["country"]
            city = address["city"]
            street = address["street"]
            postal_code = address["postal_code"]
            region = address["region"]

            return BorrowerParty(
                sequence_number=1,
                first_name=name.split(" ")[0],
                last_name=name.split(" ")[1],
                mobile_phone="9999999999",
                email=email,
                borrower_birth_date="2000-01-01",
                dependent_count="0",
                marital_status_type="Unmarried",
                current_income_monthly_total_amount="6678",
                employer_full_name="NY MTA",
                employment_start_date="2000-01-01",
                employment_time_in_line_of_work_months_count="120",
                residence_address_line_text=street,
                residence_city_name=city,
                residence_country_code=country,
                residence_postal_code=postal_code,
                residence_state_code=region,
                monthly_rent_amount="3000.00",
                borrower_residency_duration_months_count="120",
                taxpayer_identifier_value="991919991",
                borrower_residency_basis_type="Rent",
                borrower_residency_type="Current",
                citizenship_residency_type="USCitizen",
                intent_to_occupy_type="Yes",
                homeowner_past_three_years_type="No",
                employment_classification_type="Secondary",
                employment_position_description="Worker",
                employment_status_type="Current",
                party_role_type="Borrower",
            )

    
    
    def generate_property_owner_party(self) -> PropertyOwnerParty:
        return PropertyOwnerParty(
            sequence_number=1,
            property_owner_status_type="Proposed",
        )
    
    def generate_ulad(self) -> dict[str, Any]:
        assets = self.generate_assets()
        collateral = self.generate_collateral()
        liabilities = self.generate_liabilities()
        loans = self.generate_loans()
        borrower_party, property_owner_party = self.generate_parties()
        relationships = self.generate_relationships(assets, liabilities, borrower_party)
        
        ulad_payload = copy.deepcopy(self.template)
        deal = self._get_deal_node(ulad_payload)

        deal["ASSETS"]["ASSET"] = [asset.to_dict() for asset in assets]
        deal["COLLATERALS"]["COLLATERAL"] = collateral.to_dict()
        deal["LIABILITIES"]["LIABILITY"] = [
            liability.to_dict() for liability in liabilities
        ]
        deal["LOANS"]["LOAN"] = [loan.to_dict() for loan in loans]
        deal["PARTIES"]["PARTY"] = [borrower_party.to_dict(), property_owner_party.to_dict()]
        deal["RELATIONSHIPS"]["RELATIONSHIP"] = [rel.to_dict() for rel in relationships]

        if self.output_dir is not None:
            self.write_ulad(ulad_payload, self.output_dir)

        return ulad_payload

    def list_override_paths(self, leaf_only: bool = True) -> list[str]:
        """
        Enumerate valid override paths from the template.

        - leaf_only=True: return only value paths (recommended for overrides)
        - leaf_only=False: include object/list container paths too
        """
        paths: list[str] = []
        self._collect_paths(
            self.template, current_path="", paths=paths, leaf_only=leaf_only
        )
        return paths

    def write_ulad(self, ulad_payload: dict[str, Any], output_dir: str | Path) -> None:
        output_path = os.path.join(output_dir, "ulad.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ulad_payload, f, indent=2)

    def _set_value_by_path(self, obj: dict[str, Any], path: str, value: Any) -> None:
        tokens = self._parse_path(path)
        current: Any = obj

        for token in tokens[:-1]:
            if isinstance(token, int):
                current = current[token]
            else:
                current = current[token]

        final_token = tokens[-1]
        if isinstance(final_token, int):
            current[final_token] = value
        else:
            current[final_token] = value

    def _get_deal_node(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload["MESSAGE"]["DEAL_SETS"]["DEAL_SET"]["DEALS"]["DEAL"]

    def _parse_path(self, path: str) -> list[str | int]:
        """
        Convert path string into tokens.
        Example:
        "A.B[0].C" -> ["A", "B", 0, "C"]
        """
        raw_parts = path.split(".")
        tokens: list[str | int] = []

        for part in raw_parts:
            if "[" not in part:
                tokens.append(part)
                continue

            prefix = part.split("[")[0]
            if prefix:
                tokens.append(prefix)

            remainder = part[len(prefix) :]
            while remainder:
                left = remainder.index("[")
                right = remainder.index("]")
                index_text = remainder[left + 1 : right]
                tokens.append(int(index_text))
                remainder = remainder[right + 1 :]

        return tokens

    def _collect_paths(
        self,
        node: Any,
        current_path: str,
        paths: list[str],
        leaf_only: bool,
    ) -> None:
        is_container = isinstance(node, (dict, list))
        if current_path and (not leaf_only or not is_container):
            paths.append(current_path)

        if isinstance(node, dict):
            for key, value in node.items():
                next_path = f"{current_path}.{key}" if current_path else key
                self._collect_paths(value, next_path, paths, leaf_only)
            return

        if isinstance(node, list):
            for index, item in enumerate(node):
                next_path = f"{current_path}[{index}]"
                self._collect_paths(item, next_path, paths, leaf_only)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ULAD JSON from a fixed template.")
    parser.add_argument("--ulad-template", type=str, default="data/ulad_template.json", help="Input template file")
    parser.add_argument("--output-dir", type=str, default="data/test_ulad", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    generator = UladGenerator(args.ulad_template, None, args.output_dir)
    ulad = generator.generate_ulad()

    print(f"ULAD JSON generated and saved to {args.output_dir}")
