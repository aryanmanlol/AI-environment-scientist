import os
import logging
from pathlib import Path
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class ScientificKnowledgeBase:
    """
    RAG retriever module handling document loading, chunking, embedding, storage, and retrieval.
    """

    def __init__(self, data_dir: str, vectordb_dir: str, collection_name: str = 'ecointel_knowledge'):
        self.data_dir = Path(data_dir)
        self.vectordb_dir = Path(vectordb_dir)
        self.collection_name = collection_name
        
        self.chunk_size = 1000
        self.chunk_overlap = 200

        # Initialize embedding model
        model_name = "BAAI/bge-small-en-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # Initialize vector store
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.vectordb_dir)
        )

    def load_and_index_documents(self) -> int:
        """
        Load all PDFs from data_dir. If none found, load sample knowledge.
        Split documents, add to ChromaDB. Return count of chunks indexed.
        """
        try:
            if not self.data_dir.exists():
                logger.info(f"Data directory {self.data_dir} does not exist. Creating it.")
                self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Load PDFs
            logger.info(f"Loading PDFs from {self.data_dir}")
            loader = DirectoryLoader(str(self.data_dir), glob="**/*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
            
            if not documents:
                logger.info("No PDFs found. Loading built-in sample scientific knowledge.")
                documents = self._create_sample_knowledge()
                
            logger.info(f"Loaded {len(documents)} documents. Splitting text...")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            chunks = text_splitter.split_documents(documents)
            
            logger.info(f"Created {len(chunks)} chunks. Indexing into ChromaDB...")
            if chunks:
                self.vectorstore.add_documents(chunks)
                
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error loading and indexing documents: {e}")
            return 0

    def _create_sample_knowledge(self) -> List[Document]:
        """
        Create a comprehensive set of Document objects containing real scientific knowledge.
        """
        samples = [
            # 1. Soil Health and SOC
            Document(
                page_content="Soil organic carbon (SOC) is a critical indicator of soil health, influencing physical, chemical, and biological properties. Global topsoil (0-30 cm) contains approximately 680 billion tonnes of carbon. Increasing SOC through regenerative agriculture (e.g., no-till, cover cropping) enhances soil water retention, nutrient cycling, and microbial biodiversity. Research indicates that a 1% increase in SOC can increase available water holding capacity by up to 3.7% in certain soil types, providing resilience against droughts.",
                metadata={"source": "sample_soc_01", "topic": "Soil Health", "year": 2023}
            ),
            Document(
                page_content="The relationship between soil organic carbon (SOC) and biodiversity is strongly positive. High SOC levels promote a diverse and active soil microbiome, which in turn facilitates organic matter decomposition and nutrient mineralization. Conversely, intensive tillage and excessive synthetic fertilizer use typically deplete SOC and reduce microbial richness. Restoring degraded soils can sequester 0.9 to 1.85 gigatonnes of carbon globally per year, mitigating climate change while rebuilding ecosystems.",
                metadata={"source": "sample_soc_02", "topic": "Soil Health", "year": 2022}
            ),
            
            # 2. Rainfall patterns and ecosystem health
            Document(
                page_content="Changes in global rainfall patterns, characterized by an increase in extreme precipitation events interspersed with prolonged droughts, significantly impact ecosystem health. In tropical forests, extended dry seasons have been shown to increase tree mortality, particularly among large canopy trees, shifting the forest structure and reducing carbon sink capacity. Altered hydrological cycles also affect the phenology of flora, disrupting pollinator synchronization.",
                metadata={"source": "sample_rain_01", "topic": "Rainfall Patterns", "year": 2024}
            ),
            Document(
                page_content="Riparian ecosystems are particularly vulnerable to altered rainfall and snowmelt patterns. Flashier hydrology (rapid, intense runoff) leads to severe streambank erosion, sediment loading, and degradation of aquatic habitats. Maintaining robust riparian buffers with deep-rooted native vegetation is a proven strategy to mitigate these impacts, filtering runoff and stabilizing banks during high-flow events.",
                metadata={"source": "sample_rain_02", "topic": "Ecosystem Health", "year": 2021}
            ),
            
            # 3. Monoculture impacts
            Document(
                page_content="Extensive monoculture farming significantly diminishes landscape-level biodiversity (gamma diversity). By replacing complex native habitats with uniform single-crop landscapes, monocultures eliminate ecological niches required by various avian, mammalian, and insect species. Furthermore, monocultures are inherently vulnerable to pest and disease outbreaks, leading to increased reliance on broad-spectrum pesticides, which causes further collateral damage to non-target insect populations, including crucial pollinators.",
                metadata={"source": "sample_monoculture_01", "topic": "Agriculture", "year": 2023}
            ),
            Document(
                page_content="Below-ground, long-term monoculture cropping leads to a simplified soil microbial community. Plants exude specific root exudates that attract distinct microbial consortia. Without plant diversity, the microbial diversity declines, leading to 'yield decline' syndrome where the soil accumulates species-specific pathogens (e.g., specific nematodes or fungi) while losing beneficial mutualists like arbuscular mycorrhizal fungi (AMF).",
                metadata={"source": "sample_monoculture_02", "topic": "Soil Microbiology", "year": 2020}
            ),
            
            # 4. Agroforestry benefits
            Document(
                page_content="Studies by the World Agroforestry Centre (ICRAF) demonstrate that integrating trees into agricultural landscapes (agroforestry) provides multiple synergistic benefits. Trees act as windbreaks reducing soil erosion, their deep roots pump nutrients from deeper soil layers making them available to shallow-rooted crops, and they provide critical habitat for natural pest predators (e.g., insectivorous birds and parasitic wasps), thereby reducing the need for chemical pest control.",
                metadata={"source": "sample_agroforestry_01", "topic": "Agroforestry", "year": 2022}
            ),
            Document(
                page_content="Agroforestry systems are highly effective carbon sinks. Compared to treeless croplands or pastures, silvopastoral and agrosilvicultural systems can sequester significantly more carbon in above-ground biomass and soil. Additionally, the microclimate created by tree canopies buffers understory crops against extreme temperature fluctuations and excessive evapotranspiration, improving yield stability under climate stress.",
                metadata={"source": "sample_agroforestry_02", "topic": "Climate Mitigation", "year": 2021}
            ),
            
            # 5. Pollinator conservation
            Document(
                page_content="The Intergovernmental Science-Policy Platform on Biodiversity and Ecosystem Services (IPBES) reports that nearly 75% of global food crop types rely on animal pollination. The decline in wild pollinator populations (bees, butterflies, hoverflies) is driven by habitat loss, pesticide exposure (particularly neonicotinoids), and climate change. Establishing floral resource strips and preserving semi-natural habitats adjacent to agricultural fields can significantly boost wild pollinator abundance and pollination services.",
                metadata={"source": "sample_pollinator_01", "topic": "Pollinator Conservation", "year": 2023}
            ),
            Document(
                page_content="Wild pollinators often provide more efficient pollination services than managed honey bees (Apis mellifera) for certain crops. For example, bumblebees engage in 'buzz pollination' required for tomatoes and blueberries. Maintaining a diverse assemblage of pollinators ensures redundancy and resilience in pollination networks, which is crucial as phenological shifts caused by climate change threaten to decouple plant-pollinator interactions.",
                metadata={"source": "sample_pollinator_02", "topic": "Ecology", "year": 2020}
            ),

            # 6. Deforestation impacts
            Document(
                page_content="Deforestation, particularly in the Amazon and Southeast Asia, leads to catastrophic loss of species richness. The removal of tropical rainforests not only eliminates habitat for endemic species but also triggers 'extinction debts'—where isolated populations in fragmented patches slowly decline to extinction over decades due to genetic bottlenecking and edge effects.",
                metadata={"source": "sample_deforestation_01", "topic": "Biodiversity Loss", "year": 2022}
            ),
            Document(
                page_content="Beyond immediate biodiversity loss, deforestation severely disrupts local and regional climate regulation. Forests generate significant rainfall through evapotranspiration. Large-scale clearing can shift a region from a moist forest biome to a drier savanna state, a feedback loop that further stresses surviving flora and fauna and impacts agricultural viability in downwind regions.",
                metadata={"source": "sample_deforestation_02", "topic": "Climate Systems", "year": 2021}
            ),

            # 7. Cover crop benefits
            Document(
                page_content="Cover crops, particularly legumes like crimson clover and hairy vetch, fix atmospheric nitrogen (N2) through symbiotic relationships with Rhizobia bacteria. This biological nitrogen fixation can provide 50 to 150 lbs of nitrogen per acre, reducing the need for synthetic N fertilizers. The reduction in synthetic fertilizer use directly decreases emissions of nitrous oxide (N2O), a potent greenhouse gas.",
                metadata={"source": "sample_covercrop_01", "topic": "Sustainable Agriculture", "year": 2024}
            ),
            Document(
                page_content="Non-leguminous cover crops, such as cereal rye and tillage radish, excel at scavenging residual soil nitrates left after the cash crop harvest, preventing these nutrients from leaching into groundwater and causing downstream eutrophication. Furthermore, the root channels left by these cover crops when they die improve soil aeration and water infiltration.",
                metadata={"source": "sample_covercrop_02", "topic": "Soil Health", "year": 2023}
            ),

            # 8. Habitat fragmentation
            Document(
                page_content="Habitat fragmentation divides contiguous ecosystems into smaller, isolated patches. This drastically increases edge effects—changes in microclimate (temperature, light, humidity) and species composition at the boundary of a habitat. Edge-intolerant species, typical of interior old-growth forests, are often outcompeted or preyed upon by generalist 'edge' species, fundamentally altering the ecological community.",
                metadata={"source": "sample_fragmentation_01", "topic": "Landscape Ecology", "year": 2021}
            ),
            Document(
                page_content="Wildlife corridors are essential mitigations for habitat fragmentation. By physically connecting isolated patches, corridors facilitate gene flow, seasonal migrations, and range shifts necessary for species adapting to climate change. The design of effective corridors requires understanding the specific mobility needs and dispersal behaviors of the target species.",
                metadata={"source": "sample_fragmentation_02", "topic": "Conservation Biology", "year": 2022}
            ),

            # 9. Climate change impacts (IPCC)
            Document(
                page_content="According to the Intergovernmental Panel on Climate Change (IPCC), global warming exceeding 1.5°C poses severe risks to terrestrial and marine biodiversity. Coral reefs are exceptionally vulnerable, with warming and ocean acidification causing widespread bleaching events and structural degradation. Up to 90% of warm-water coral reefs are projected to suffer severe degradation even at 1.5°C warming.",
                metadata={"source": "sample_ipcc_01", "topic": "Climate Change", "year": 2023}
            ),
            Document(
                page_content="Climate change is driving shifts in species distributions, generally moving them poleward or to higher elevations to track suitable climate envelopes. However, species with limited mobility, or those restricted by topographical barriers (e.g., species already at mountain summits), face high extinction risks as their habitable range shrinks and disappears.",
                metadata={"source": "sample_ipcc_02", "topic": "Climate Ecology", "year": 2022}
            ),

            # 10. Water quality and aquatic biodiversity
            Document(
                page_content="Nutrient pollution (nitrogen and phosphorus) from agricultural runoff and wastewater discharge is a primary driver of freshwater biodiversity loss. Excessive nutrients trigger algal blooms, which deplete dissolved oxygen in the water column when they decompose, leading to hypoxic 'dead zones' where most fish and benthic macroinvertebrates cannot survive.",
                metadata={"source": "sample_water_01", "topic": "Aquatic Ecosystems", "year": 2021}
            ),
            Document(
                page_content="The presence and diversity of benthic macroinvertebrates (e.g., mayflies, stoneflies, caddisflies) serve as robust bioindicators of stream health. These taxa exhibit varying sensitivities to chemical pollutants, sedimentation, and thermal pollution. A shift from a community dominated by sensitive species to one dominated by tolerant species (e.g., chironomids, leeches) indicates degraded water quality.",
                metadata={"source": "sample_water_02", "topic": "Bioindication", "year": 2020}
            ),

            # 11. Soil microbiome and plant health
            Document(
                page_content="The plant rhizosphere—the narrow region of soil directly influenced by root secretions—hosts a complex microbiome that acts as a secondary immune system for plants. Beneficial microbes, such as Plant Growth-Promoting Rhizobacteria (PGPR), can induce systemic resistance in the plant, priming it to defend against foliar and root pathogens more rapidly and intensely.",
                metadata={"source": "sample_microbiome_01", "topic": "Plant Pathology", "year": 2023}
            ),
            Document(
                page_content="Mycorrhizal networks, formed by symbiotic fungi connecting multiple plant root systems, facilitate resource sharing in plant communities. These common mycorrhizal networks (CMNs) can transfer carbon, nitrogen, and phosphorus between plants. Furthermore, when a plant is attacked by herbivores, warning chemical signals can travel through the CMN, prompting neighboring plants to upregulate their chemical defenses.",
                metadata={"source": "sample_microbiome_02", "topic": "Symbiosis", "year": 2024}
            ),

            # 12. Urban ecology
            Document(
                page_content="Urban green infrastructure (UGI), including parks, green roofs, and street trees, plays a vital role in mitigating the urban heat island effect and managing stormwater runoff. Properly designed UGI can also support surprising levels of biodiversity, providing critical stopover habitats for migratory birds and refuges for urban-adapted pollinators.",
                metadata={"source": "sample_urban_01", "topic": "Urban Ecology", "year": 2022}
            ),
            Document(
                page_content="The concept of 'reconciliation ecology' argues for the deliberate design of human-dominated landscapes to accommodate wild species. Examples include modifying building designs to incorporate bat roosts or bird nesting cavities, and replacing manicured lawns with native meadow plantings. This approach is essential because traditional protected areas alone are insufficient to halt global biodiversity loss.",
                metadata={"source": "sample_urban_02", "topic": "Reconciliation Ecology", "year": 2021}
            )
        ]
        return samples

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve top-k relevant documents using similarity search.
        """
        try:
            logger.info(f"Retrieving {k} documents for query: {query}")
            docs = self.vectorstore.similarity_search(query, k=k)
            return docs
        except Exception as e:
            logger.error(f"Error in retrieve: {e}")
            return []

    def multi_query_retrieve(self, queries: List[str], k: int = 3) -> List[Document]:
        """
        Execute multiple queries, deduplicate results, and return unique documents.
        """
        try:
            logger.info(f"Executing multi-query retrieval for {len(queries)} queries.")
            all_docs = []
            
            for query in queries:
                docs = self.vectorstore.similarity_search(query, k=k)
                all_docs.extend(docs)
                
            # Deduplicate by page_content
            unique_docs = []
            seen_contents = set()
            
            for doc in all_docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    unique_docs.append(doc)
                    
            logger.info(f"Retrieved {len(unique_docs)} unique documents from multi-query.")
            return unique_docs
        except Exception as e:
            logger.error(f"Error in multi_query_retrieve: {e}")
            return []

    def get_collection_stats(self) -> dict:
        """
        Return stats about the vector DB.
        """
        try:
            # Chroma DB get() returns dict with 'ids', 'documents', 'metadatas' etc.
            collection_data = self.vectorstore.get()
            count = len(collection_data.get('ids', []))
            return {
                "collection_name": self.collection_name,
                "count": count
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "collection_name": self.collection_name,
                "count": 0,
                "error": str(e)
            }
