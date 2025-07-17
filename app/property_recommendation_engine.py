# /app/property_recommendation_engine.py
"""
Property Recommendation Engine for Smart Property Recommendations

This service generates personalized property recommendations using the existing RAG
infrastructure, ranking and filtering results based on user context and preferences.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from llama_index.core.llms import ChatMessage
from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore

from app.models import PropertyRecommendation, UserContext, RecommendationResult
import app.rag as rag_module

logger = logging.getLogger(__name__)


class PropertyRecommendationEngine:
    """
    Service for generating personalized property recommendations using RAG and user context.
    
    Leverages the existing ChromaDB and hybrid search infrastructure to find and rank
    properties based on user preferences and context.
    """
    
    # Scoring weights for different matching criteria
    SCORING_WEIGHTS = {
        'budget_match': 0.30,
        'location_match': 0.25,
        'feature_match': 0.20,
        'size_match': 0.15,
        'property_type_match': 0.10
    }
    
    # Default number of recommendations to return
    DEFAULT_MAX_RESULTS = 3
    
    def __init__(self):
        pass
    
    async def generate_recommendations(self, user_context: UserContext, 
                                     max_results: int = DEFAULT_MAX_RESULTS) -> List[PropertyRecommendation]:
        """
        Generate personalized property recommendations for a user.
        
        Args:
            user_context: User's context with preferences and history
            max_results: Maximum number of recommendations to return
            
        Returns:
            List of PropertyRecommendation objects
        """
        logger.info(f"Generating recommendations for user {user_context.user_id} (max: {max_results})")
        
        # Step 1: Build search queries based on user preferences
        search_queries = self._build_search_queries(user_context)
        
        # Step 2: Retrieve properties using RAG
        all_properties = await self._retrieve_properties(user_context.user_id, search_queries)
        
        # Step 3: Score and rank properties
        scored_properties = self._score_properties(all_properties, user_context)
        
        # Step 4: Filter and select top recommendations
        recommendations = await self._select_top_recommendations(
            scored_properties, 
            user_context, 
            max_results
        )
        
        logger.info(f"Generated {len(recommendations)} recommendations for user {user_context.user_id}")
        return recommendations
    
    async def explain_recommendation(self, property_rec: PropertyRecommendation, 
                                   user_context: UserContext) -> str:
        """
        Generate a detailed explanation for why a property was recommended.
        
        Args:
            property_rec: Property recommendation to explain
            user_context: User's context and preferences
            
        Returns:
            Detailed explanation string
        """
        explanation_prompt = f"""
        Generate a personalized explanation for why this property is recommended to the user.
        
        Property Details:
        {json.dumps(property_rec.property_data, indent=2)}
        
        User Preferences:
        Budget Range: {user_context.budget_range}
        Preferred Locations: {user_context.preferred_locations}
        Required Features: {user_context.required_features}
        Historical Preferences: {json.dumps(user_context.historical_preferences, indent=2)}
        
        Matching Criteria: {property_rec.matching_criteria}
        Match Score: {property_rec.match_score:.2f}
        
        Create a brief, personalized explanation (2-3 sentences) that:
        1. Highlights the key features that match the user's preferences
        2. Mentions specific details from their history if relevant
        3. Explains why this property is a good fit
        4. Sounds natural and conversational
        
        Example: "This apartment has the chef's kitchen you mentioned wanting, and it's in your preferred downtown area. At $2,800/month, it fits perfectly within your budget range, and the building includes the gym amenities you're looking for."
        
        Generate the explanation:
        """
        
        try:
            response = await Settings.llm.achat([ChatMessage(role="user", content=explanation_prompt)])
            explanation = response.message.content.strip() if response.message.content else ""
            
            # Validate and clean the explanation
            if explanation and len(explanation) > 20 and not explanation.startswith('{'):
                return explanation
            else:
                # Fallback explanation
                return self._generate_fallback_explanation(property_rec, user_context)
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return self._generate_fallback_explanation(property_rec, user_context)
    
    def _build_search_queries(self, user_context: UserContext) -> List[str]:
        """Build search queries based on user preferences."""
        queries = []
        
        # Build queries for different aspects of user preferences
        
        # Budget-based query
        if user_context.budget_range:
            min_budget, max_budget = user_context.budget_range
            queries.append(f"properties rent between ${min_budget} and ${max_budget}")
        
        # Location-based queries
        for location in user_context.preferred_locations:
            queries.append(f"properties in {location}")
            queries.append(f"apartments near {location}")
        
        # Feature-based queries
        for feature in user_context.required_features:
            queries.append(f"properties with {feature}")
            queries.append(f"apartments {feature}")
        
        # Historical preference queries
        hist_prefs = user_context.historical_preferences
        if hist_prefs.get('property_type'):
            prop_type = hist_prefs['property_type']
            queries.append(f"{prop_type} properties")
        
        if hist_prefs.get('size', {}).get('bedrooms'):
            bedrooms = hist_prefs['size']['bedrooms']
            queries.append(f"{bedrooms} bedroom apartments")
        
        # General fallback queries
        if not queries:
            queries.extend([
                "available apartments",
                "rental properties",
                "property listings"
            ])
        
        # Limit number of queries to avoid overloading
        return queries[:8]
    
    async def _retrieve_properties(self, user_id: str, queries: List[str]) -> List[NodeWithScore]:
        """Retrieve properties using the existing RAG system."""
        all_nodes = []
        seen_property_ids = set()
        
        for query in queries:
            try:
                # Use the existing multi-strategy search
                search_result = await rag_module.retrieve_context_optimized(query, user_id)
                
                # Collect unique nodes
                for node in search_result.nodes_found:
                    # Create a unique ID for the property to avoid duplicates
                    property_id = self._extract_property_id(node)
                    if property_id not in seen_property_ids:
                        all_nodes.append(node)
                        seen_property_ids.add(property_id)
                
            except Exception as e:
                logger.error(f"Error retrieving properties for query '{query}': {e}")
                continue
        
        logger.info(f"Retrieved {len(all_nodes)} unique properties from {len(queries)} queries")
        return all_nodes
    
    def _extract_property_id(self, node: NodeWithScore) -> str:
        """Extract a unique property identifier from a node."""
        # Try to get property address from metadata or content
        if hasattr(node, 'metadata') and node.metadata:
            address = node.metadata.get('property address')
            if address:
                return str(address).strip().lower()
        
        # Fallback: extract address from content
        content = node.get_content()
        address_match = re.search(r'property address:\s*([^,]+)', content, re.IGNORECASE)
        if address_match:
            return address_match.group(1).strip().lower()
        
        # Last resort: use node ID or content hash
        return getattr(node, 'id_', str(hash(content))[:10])
    
    def _score_properties(self, properties: List[NodeWithScore], 
                         user_context: UserContext) -> List[Tuple[NodeWithScore, float, Dict[str, Any]]]:
        """Score properties based on how well they match user preferences."""
        scored_properties = []
        
        for node in properties:
            # Extract property data from node
            property_data = self._extract_property_data(node)
            
            # Calculate match scores for different criteria
            budget_score = self._calculate_budget_match(property_data, user_context)
            location_score = self._calculate_location_match(property_data, user_context)
            feature_score = self._calculate_feature_match(property_data, user_context)
            size_score = self._calculate_size_match(property_data, user_context)
            type_score = self._calculate_type_match(property_data, user_context)
            
            # Calculate weighted total score
            total_score = (
                budget_score * self.SCORING_WEIGHTS['budget_match'] +
                location_score * self.SCORING_WEIGHTS['location_match'] +
                feature_score * self.SCORING_WEIGHTS['feature_match'] +
                size_score * self.SCORING_WEIGHTS['size_match'] +
                type_score * self.SCORING_WEIGHTS['property_type_match']
            )
            
            # Create match details
            match_details = {
                'budget_score': budget_score,
                'location_score': location_score,
                'feature_score': feature_score,
                'size_score': size_score,
                'type_score': type_score,
                'total_score': total_score
            }
            
            scored_properties.append((node, total_score, match_details))
        
        # Sort by score (highest first)
        scored_properties.sort(key=lambda x: x[1], reverse=True)
        
        return scored_properties
    
    def _extract_property_data(self, node: NodeWithScore) -> Dict[str, Any]:
        """Extract structured property data from a node."""
        property_data = {}
        
        # Get data from metadata if available
        if hasattr(node, 'metadata') and node.metadata:
            property_data.update(node.metadata)
        
        # Parse content for additional data
        content = node.get_content()
        
        # Extract key property information using regex
        patterns = {
            'property_address': r'property address:\s*([^,\n]+)',
            'monthly_rent': r'monthly rent:\s*\$?([0-9,]+)',
            'size_sf': r'size \(sf\):\s*([0-9,]+)',
            'property_type': r'property type:\s*([^,\n]+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match and key not in property_data:
                value = match.group(1).strip()
                # Convert numeric values
                if key in ['monthly_rent', 'size_sf']:
                    try:
                        property_data[key] = int(value.replace(',', ''))
                    except ValueError:
                        property_data[key] = value
                else:
                    property_data[key] = value
        
        return property_data
    
    def _calculate_budget_match(self, property_data: Dict[str, Any], 
                               user_context: UserContext) -> float:
        """Calculate how well the property matches user's budget."""
        if not user_context.budget_range:
            return 0.5  # Neutral score if no budget preference
        
        rent = property_data.get('monthly_rent') or property_data.get('monthly rent')
        if not rent:
            return 0.3  # Lower score if rent is unknown
        
        try:
            rent = int(str(rent).replace(',', '').replace('$', ''))
        except (ValueError, TypeError):
            return 0.3
        
        min_budget, max_budget = user_context.budget_range
        
        # Perfect match if within range
        if min_budget <= rent <= max_budget:
            return 1.0
        
        # Partial match if close to range
        if rent < min_budget:
            # Below budget is good, but prefer closer to minimum
            ratio = rent / min_budget
            return max(0.7, ratio)
        else:
            # Above budget is less desirable
            if rent <= max_budget * 1.1:  # Within 10% over budget
                return 0.6
            elif rent <= max_budget * 1.2:  # Within 20% over budget
                return 0.4
            else:
                return 0.1  # Way over budget
    
    def _calculate_location_match(self, property_data: Dict[str, Any], 
                                 user_context: UserContext) -> float:
        """Calculate how well the property matches user's location preferences."""
        if not user_context.preferred_locations:
            return 0.5  # Neutral if no location preference
        
        property_address = (
            property_data.get('property_address') or 
            property_data.get('property address', '')
        ).lower()
        
        if not property_address:
            return 0.3  # Lower score if address is unknown
        
        # Check for matches with preferred locations
        max_score = 0.0
        for location in user_context.preferred_locations:
            location_lower = location.lower()
            
            # Exact match
            if location_lower in property_address:
                max_score = max(max_score, 1.0)
            # Partial match (e.g., "downtown" matches "downtown manhattan")
            elif any(word in property_address for word in location_lower.split()):
                max_score = max(max_score, 0.8)
        
        return max_score
    
    def _calculate_feature_match(self, property_data: Dict[str, Any], 
                                user_context: UserContext) -> float:
        """Calculate how well the property matches user's feature requirements."""
        if not user_context.required_features:
            return 0.5  # Neutral if no feature requirements
        
        # Get all text content to search for features
        all_content = ' '.join([
            str(v).lower() for v in property_data.values() if v
        ])
        
        matched_features = 0
        total_features = len(user_context.required_features)
        
        for feature in user_context.required_features:
            feature_lower = feature.lower()
            
            # Check for feature mentions in property data
            if feature_lower in all_content:
                matched_features += 1
            # Check for partial matches (e.g., "gym" in "fitness center")
            elif any(word in all_content for word in feature_lower.split()):
                matched_features += 0.5
        
        return min(1.0, matched_features / total_features)
    
    def _calculate_size_match(self, property_data: Dict[str, Any], 
                             user_context: UserContext) -> float:
        """Calculate how well the property matches user's size requirements."""
        hist_prefs = user_context.historical_preferences
        size_prefs = hist_prefs.get('size', {})
        
        if not size_prefs:
            return 0.5  # Neutral if no size preference
        
        # Check square footage
        if 'min_sqft' in size_prefs:
            min_sqft = size_prefs['min_sqft']
            property_sqft = property_data.get('size_sf') or property_data.get('size (sf)')
            
            if property_sqft:
                try:
                    property_sqft = int(str(property_sqft).replace(',', ''))
                    if property_sqft >= min_sqft:
                        return 1.0
                    elif property_sqft >= min_sqft * 0.8:  # Within 20% of requirement
                        return 0.7
                    else:
                        return 0.3
                except (ValueError, TypeError):
                    pass
        
        # TODO: Add bedroom/bathroom matching when available in data
        
        return 0.5  # Default neutral score
    
    def _calculate_type_match(self, property_data: Dict[str, Any], 
                             user_context: UserContext) -> float:
        """Calculate how well the property matches user's property type preference."""
        hist_prefs = user_context.historical_preferences
        preferred_type = hist_prefs.get('property_type')
        
        if not preferred_type:
            return 0.5  # Neutral if no type preference
        
        property_type = (
            property_data.get('property_type') or 
            property_data.get('property type', '')
        ).lower()
        
        if not property_type:
            return 0.4  # Lower score if type is unknown
        
        preferred_type_lower = preferred_type.lower()
        
        # Exact match
        if preferred_type_lower == property_type:
            return 1.0
        # Partial match
        elif preferred_type_lower in property_type or property_type in preferred_type_lower:
            return 0.8
        else:
            return 0.2
    
    async def _select_top_recommendations(self, scored_properties: List[Tuple[NodeWithScore, float, Dict[str, Any]]], 
                                        user_context: UserContext, 
                                        max_results: int) -> List[PropertyRecommendation]:
        """Select and format the top property recommendations."""
        recommendations = []
        
        for i, (node, score, match_details) in enumerate(scored_properties[:max_results]):
            property_data = self._extract_property_data(node)
            property_id = self._extract_property_id(node)
            
            # Generate matching criteria description
            matching_criteria = self._generate_matching_criteria(match_details, user_context)
            
            # Create basic explanation (will be enhanced by explain_recommendation if needed)
            explanation = await self.explain_recommendation(
                PropertyRecommendation(
                    property_id=property_id,
                    property_data=property_data,
                    match_score=score,
                    explanation="",  # Will be filled
                    matching_criteria=matching_criteria
                ),
                user_context
            )
            
            recommendation = PropertyRecommendation(
                property_id=property_id,
                property_data=property_data,
                match_score=score,
                explanation=explanation,
                matching_criteria=matching_criteria
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_matching_criteria(self, match_details: Dict[str, Any], 
                                   user_context: UserContext) -> List[str]:
        """Generate a list of matching criteria for the property."""
        criteria = []
        
        # Budget match
        if match_details['budget_score'] > 0.7:
            criteria.append("within budget")
        
        # Location match
        if match_details['location_score'] > 0.7:
            criteria.append("preferred location")
        
        # Feature match
        if match_details['feature_score'] > 0.6:
            criteria.append("has required amenities")
        
        # Size match
        if match_details['size_score'] > 0.7:
            criteria.append("meets size requirements")
        
        # Type match
        if match_details['type_score'] > 0.7:
            criteria.append("preferred property type")
        
        return criteria if criteria else ["general match"]
    
    def _generate_fallback_explanation(self, property_rec: PropertyRecommendation, 
                                     user_context: UserContext) -> str:
        """Generate a fallback explanation when LLM explanation fails."""
        address = property_rec.property_data.get('property_address', 'This property')
        rent = property_rec.property_data.get('monthly_rent')
        
        explanation = f"{address} is recommended for you"
        
        if rent and user_context.budget_range:
            min_budget, max_budget = user_context.budget_range
            if min_budget <= rent <= max_budget:
                explanation += f" because it's within your budget at ${rent:,}/month"
        
        if property_rec.matching_criteria:
            criteria_text = ", ".join(property_rec.matching_criteria)
            explanation += f" and matches your preferences for {criteria_text}"
        
        explanation += "."
        
        return explanation


# Global service instance
property_recommendation_engine = PropertyRecommendationEngine() 