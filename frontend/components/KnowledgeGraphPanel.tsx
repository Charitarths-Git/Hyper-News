"use client";

import dynamic from "next/dynamic";
import { Maximize2, Minimize2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { apiUrl } from "@/lib/api";
import { Pill, StateBlock, SurfaceCard } from "@/components/ui/primitives";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
  loading: () => (
    <div className="flex h-[350px] items-center justify-center text-sm text-[var(--muted)]">
      Loading graph...
    </div>
  ),
});

interface GraphNode {
  id: string;
  label: string;
  type: string;
  degree?: number;
}

interface GraphLink {
  source: string;
  target: string;
  relation?: string;
}

interface GraphEntity {
  id: string;
  type: string;
  connections: number;
}

interface GraphResponse {
  error?: string;
  top_entities?: GraphEntity[];
  nodes?: GraphNode[];
  links?: GraphLink[];
}

interface GraphData {
  nodes: Array<GraphNode & { val: number; name: string }>;
  links: GraphLink[];
}

const TYPE_COLORS: Record<string, string> = {
  article: "#64748b",
  category: "#e6b86c",
  subcategory: "#f97316",
  PERSON: "#10b981",
  GPE: "#f59e0b",
  ORG: "#3b82f6",
  CONCEPT: "#8b5cf6",
  ENTITY: "#94a3b8",
  EVENT: "#ec4899",
  LOCATION: "#06b6d4",
};

export default function KnowledgeGraphPanel() {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [topEntities, setTopEntities] = useState<GraphEntity[]>([]);
  const [dimensions, setDimensions] = useState({ width: 800, height: 360 });
  const [isExpanded, setIsExpanded] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch(apiUrl("/graph"))
      .then((response) => response.json())
      .then((data: GraphResponse) => {
        if (data.error || !data.nodes?.length) {
          return;
        }
        setTopEntities(data.top_entities ?? []);
        setGraphData({
          nodes: data.nodes.map((node) => ({
            ...node,
            name: node.label,
            val:
              node.type === "article"
                ? 2.5
                : Math.max(
                    3,
                    Math.min(12, (node.degree ?? 1) / (node.type === "category" ? 1.3 : 1.8)),
                  ),
          })),
          links: data.links ?? [],
        });
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const update = () => {
      if (!containerRef.current) {
        return;
      }
      setDimensions({
        width: containerRef.current.clientWidth,
        height: isExpanded ? 620 : 360,
      });
    };

    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, [isExpanded]);

  const getNodeColor = useCallback((node: unknown) => {
    const parsed = node as Partial<GraphNode>;
    return TYPE_COLORS[String(parsed.type ?? "")] ?? "#94a3b8";
  }, []);

  if (!graphData || graphData.nodes.length === 0) {
    return (
      <SurfaceCard className="space-y-4">
        <h3 className="text-lg font-semibold text-[var(--foreground)]">Knowledge graph</h3>
        <StateBlock
          title="Graph unavailable"
          description="No entity graph data available yet. Read more articles to build graph connections."
          tone="warning"
        />
      </SurfaceCard>
    );
  }

  return (
    <SurfaceCard className="space-y-4" ref={containerRef}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-lg font-semibold text-[var(--foreground)]">Knowledge graph</h3>
        <button
          onClick={() => setIsExpanded((current) => !current)}
          className="rounded-full border border-white/12 bg-[linear-gradient(180deg,rgba(255,255,255,0.12),rgba(255,255,255,0.04))] px-4 py-2 text-xs uppercase tracking-[0.18em] text-[var(--muted-strong)] backdrop-blur-xl"
        >
          {isExpanded ? (
            <Minimize2 size={14} className="mr-1 inline" />
          ) : (
            <Maximize2 size={14} className="mr-1 inline" />
          )}
          {isExpanded ? "Collapse" : "Expand"}
        </button>
      </div>

      <div className="overflow-hidden rounded-[20px] border border-white/10 bg-[radial-gradient(circle_at_center,rgba(230,184,108,0.08),transparent_60%),linear-gradient(180deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02))]">
        <ForceGraph2D
          width={Math.max(320, dimensions.width - 40)}
          height={dimensions.height}
          graphData={graphData}
          nodeLabel={(node) => {
            const parsed = node as Partial<GraphNode>;
            return `${parsed.label ?? parsed.id} (${parsed.type ?? "entity"})`;
          }}
          nodeColor={getNodeColor}
          nodeRelSize={5}
          linkColor={() => "rgba(255,255,255,0.11)"}
          backgroundColor="transparent"
          d3AlphaDecay={0.035}
          d3VelocityDecay={0.3}
          cooldownTicks={90}
        />
      </div>

      <div className="flex flex-wrap gap-3 text-xs text-[var(--muted)]">
        {[
          { color: TYPE_COLORS.category, label: "Category" },
          { color: TYPE_COLORS.article, label: "Article" },
          { color: TYPE_COLORS.PERSON, label: "Person" },
          { color: TYPE_COLORS.ORG, label: "Organization" },
          { color: TYPE_COLORS.GPE, label: "Location" },
          { color: TYPE_COLORS.CONCEPT, label: "Concept" },
        ].map(({ color, label }) => (
          <div key={label} className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: color }} />
            {label}
          </div>
        ))}
      </div>

      {topEntities.length > 0 && (
        <div className="space-y-3">
          <p className="text-xs uppercase tracking-[0.2em] text-[var(--muted)]">Top entities</p>
          <div className="flex flex-wrap gap-2">
            {topEntities.slice(0, 10).map((entity, index) => (
              <Pill key={`${entity.id}-${entity.type}`} active={index === 0}>
                {entity.id} | {entity.connections}
              </Pill>
            ))}
          </div>
        </div>
      )}
    </SurfaceCard>
  );
}
